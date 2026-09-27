"""Live flow ingestion.

`POST /api/ingest` accepts batches of flow records from an external collector.
Each source keeps a rolling buffer; whenever a full window's worth of flows has
arrived, the window state is computed, appended to a rolling context of the
last L states, and scored. Results are pushed to any dashboard subscribed to
that source.

This is the seam a production deployment would attach a NetFlow/IPFIX
collector, a Zeek `conn.log` tail or a Kafka consumer to: everything
downstream of `LiveSource.add_flows` is identical whether the flows came from
a replayed CSV or a live tap.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from netforecast.features import N_FEATURES, compute_window_state, missing_columns, window_label
from netforecast.mitre_mapping import STAGE_TO_ID
from netforecast.predict import InfiltrationPredictor


@dataclass
class LiveWindow:
    seq: int
    t: float
    n_flows: int
    probability: float
    stage: str
    horizon: list
    overall_risk: float
    truth_stage: Optional[str] = None


@dataclass
class LiveSource:
    """Rolling per-source ingest state.

    `mode` decides what closes a window. It must match how the active model
    was trained, or the state vectors are computed over the wrong aggregation
    and every prediction is meaningless:

    * ``"time"``  -- windows are `window_size` seconds of wall clock, bucketed
      on ``timestamp // window_size``. This is how the synthetic model was
      trained.
    * ``"count"`` -- windows are `window_size` consecutive flows. CIC-IDS2017
      has no usable clock, so its adapter uses row order and the model's
      "window_seconds: 200" is really 200 flows.
    """
    source_id: str
    window_size: int
    mode: str
    context_len: int
    horizon_k: int
    buffer: list = field(default_factory=list)
    current_window_id: Optional[int] = None
    context: deque = field(default_factory=deque)
    history: list = field(default_factory=list)
    seq: int = 0
    total_flows: int = 0
    created_at: float = field(default_factory=time.time)
    subscribers: set = field(default_factory=set)

    def snapshot(self) -> dict:
        return {
            "source_id": self.source_id,
            "windows_completed": self.seq,
            "total_flows": self.total_flows,
            "buffered_flows": len(self.buffer),
            "window_size": self.window_size,
            "mode": self.mode,
            "context_filled": len(self.context),
            "context_len": self.context_len,
            "created_at": self.created_at,
            "recent": [w.__dict__ for w in self.history[-200:]],
        }


class IngestEngine:
    def __init__(self, predictor_factory, mode: str = "time"):
        self._predictor_factory = predictor_factory
        self._mode = mode
        self.sources: dict[str, LiveSource] = {}
        self._lock = asyncio.Lock()

    def _predictor(self) -> InfiltrationPredictor:
        return self._predictor_factory()

    def source(self, source_id: str, window_size: Optional[int] = None,
               mode: Optional[str] = None) -> LiveSource:
        if source_id not in self.sources:
            p = self._predictor()
            self.sources[source_id] = LiveSource(
                source_id=source_id,
                window_size=int(window_size or p.cfg["window_seconds"]),
                mode=mode or self._mode,
                context_len=int(p.cfg["context_len"]),
                horizon_k=int(p.cfg["horizon_k"]),
            )
        return self.sources[source_id]

    def reset(self, source_id: str) -> bool:
        return self.sources.pop(source_id, None) is not None

    async def add_flows(self, source_id: str, flows: list,
                        window_size: Optional[int] = None,
                        mode: Optional[str] = None) -> dict:
        """Buffer incoming flows, emitting a scored window each time one closes."""
        if not flows:
            return {"accepted": 0, "windows": []}

        df = pd.DataFrame(flows)
        missing = missing_columns(df)
        if missing:
            raise ValueError("flow records are missing required column(s): " + ", ".join(missing))

        async with self._lock:
            src = self.source(source_id, window_size, mode)
            src.total_flows += len(df)
            emitted = (self._add_by_time(src, df) if src.mode == "time"
                       else self._add_by_count(src, df))

        for w in emitted:
            await self._broadcast(src, w)
        return {"accepted": len(df), "windows": [w.__dict__ for w in emitted],
                "buffered": len(src.buffer)}

    def _add_by_count(self, src: LiveSource, df: pd.DataFrame) -> list:
        src.buffer.extend(df.to_dict(orient="records"))
        emitted = []
        while len(src.buffer) >= src.window_size:
            chunk, src.buffer = src.buffer[:src.window_size], src.buffer[src.window_size:]
            emitted.append(self._score_window(src, pd.DataFrame(chunk)))
        return emitted

    def _add_by_time(self, src: LiveSource, df: pd.DataFrame) -> list:
        """Close a window as soon as a flow from a later time bucket arrives.

        Empty intervening buckets are emitted as zero states, so a quiet period
        advances the model's context exactly as it does during training rather
        than being silently skipped.
        """
        emitted = []
        wid_all = (pd.to_numeric(df["timestamp"], errors="coerce")
                   .fillna(0) // src.window_size).astype("int64").to_numpy()
        records = df.to_dict(orient="records")

        for wid, rec in zip(wid_all, records):
            wid = int(wid)
            if src.current_window_id is None:
                src.current_window_id = wid
            elif wid > src.current_window_id:
                if src.buffer:
                    emitted.append(self._score_window(src, pd.DataFrame(src.buffer)))
                    src.buffer = []
                for gap in range(src.current_window_id + 1, wid):
                    emitted.append(self._score_window(src, None, empty_at=gap))
                src.current_window_id = wid
            src.buffer.append(rec)
        return emitted

    async def flush(self, source_id: str) -> dict:
        """Score whatever is left in the buffer as a final partial window."""
        async with self._lock:
            src = self.sources.get(source_id)
            if not src or not src.buffer:
                return {"windows": []}
            chunk, src.buffer = src.buffer, []
            w = self._score_window(src, pd.DataFrame(chunk))
        await self._broadcast(src, w)
        return {"windows": [w.__dict__]}

    def _score_window(self, src: LiveSource, win: Optional[pd.DataFrame],
                      empty_at: Optional[int] = None) -> LiveWindow:
        predictor = self._predictor()
        empty = win is None or len(win) == 0
        state = np.zeros(N_FEATURES, dtype=np.float32) if empty else compute_window_state(win)
        norm = predictor.normalize(state.reshape(1, N_FEATURES)).astype(np.float32)[0]

        src.context.append(norm)
        while len(src.context) > src.context_len:
            src.context.popleft()

        ctx = np.stack(list(src.context))
        result = predictor.rollout(ctx, k=src.horizon_k, explain=False)
        first = result["forecast"][0]

        truth = None
        if not empty and "label_stage" in win.columns:
            truth = window_label(win)

        if empty:
            t = float((empty_at if empty_at is not None else src.seq) * src.window_size)
        elif "timestamp" in win.columns:
            t = float(win["timestamp"].iloc[-1])
        else:
            t = float(src.seq)

        src.seq += 1
        lw = LiveWindow(
            seq=src.seq,
            t=t,
            n_flows=0 if empty else int(len(win)),
            probability=float(first["infiltration_probability"]),
            stage=first["predicted_stage"],
            horizon=[{"step": s["step"],
                      "probability": round(float(s["infiltration_probability"]), 5),
                      "stage": s["predicted_stage"]} for s in result["forecast"]],
            overall_risk=float(result["overall_infiltration_risk"]),
            truth_stage=truth,
        )
        src.history.append(lw)
        if len(src.history) > 5000:
            del src.history[:1000]
        return lw

    async def _broadcast(self, src: LiveSource, w: LiveWindow) -> None:
        if not src.subscribers:
            return
        payload = {"type": "live_window", "source_id": src.source_id, **w.__dict__}
        dead = set()
        for ws in list(src.subscribers):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        src.subscribers -= dead

    def subscribe(self, source_id: str, ws) -> LiveSource:
        src = self.source(source_id)
        src.subscribers.add(ws)
        return src

    def unsubscribe(self, source_id: str, ws) -> None:
        src = self.sources.get(source_id)
        if src:
            src.subscribers.discard(ws)


__all__ = ["IngestEngine", "LiveSource", "LiveWindow", "STAGE_TO_ID"]
