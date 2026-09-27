"""Session store: loaded captures, their windowed state sequences and risk timelines.

This is what replaced Streamlit's `@st.cache_data`. The important property is
that a session never holds the ingested DataFrame: it keeps the state matrix,
a compact :class:`~netforecast.features.WindowIndex`, and a downcast display
frame of only the columns the flow panels render. A CIC-IDS2017 session costs
~80 MB resident instead of ~1.0 GB, and nothing is deep-copied per request.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from netforecast.features import WindowIndex, build_state_sequence, missing_columns
from netforecast.mitre_mapping import ID_TO_STAGE
from netforecast.predict import InfiltrationPredictor

from .config import DISPLAY_COLUMNS, TIMELINE_MAX_POINTS, DatasetSpec


class SchemaError(ValueError):
    """Uploaded flow records do not carry the required columns."""

    def __init__(self, missing: list):
        self.missing = missing
        super().__init__("flow records are missing required column(s): " + ", ".join(missing))


@dataclass
class LoadProgress:
    stage: str = "queued"
    pct: float = 0.0
    message: str = ""
    error: Optional[str] = None
    done: bool = False


@dataclass
class Session:
    id: str
    dataset_id: str
    label: str
    source: str
    window_seconds: int
    window_unit: str
    context_len: int
    horizon_k: int
    n_rows: int = 0
    states: Optional[np.ndarray] = None
    states_norm: Optional[np.ndarray] = None
    timestamps: Optional[np.ndarray] = None
    label_ids: Optional[np.ndarray] = None
    window_index: Optional[WindowIndex] = None
    display: Optional[pd.DataFrame] = None
    timeline: Optional[dict] = None
    progress: LoadProgress = field(default_factory=LoadProgress)
    created_at: float = field(default_factory=time.time)

    @property
    def n_windows(self) -> int:
        return 0 if self.states is None else int(len(self.states))

    def flows_for_window(self, window: int, limit: int = 50) -> list:
        """Rows belonging to a window, sliced lazily out of the display frame."""
        if self.display is None or self.window_index is None:
            return []
        if not (0 <= window < len(self.window_index)):
            return []
        rows = self.window_index.rows_for(window, limit=limit)
        if len(rows) == 0:
            return []
        sub = self.display.iloc[rows]
        return [
            {k: (None if pd.isna(v) else (v.item() if hasattr(v, "item") else v))
             for k, v in rec.items()}
            for rec in sub.to_dict(orient="records")
        ]

    def window_flow_count(self, window: int) -> int:
        if self.window_index is None or not (0 <= window < len(self.window_index)):
            return 0
        return self.window_index.count_for(window)


def _downcast_display(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DISPLAY_COLUMNS if c in df.columns]
    out = df[cols].copy()
    for c in out.columns:
        if out[c].dtype == object:
            out[c] = out[c].astype("category")
        elif str(out[c].dtype).startswith("int"):
            out[c] = pd.to_numeric(out[c], downcast="integer")
        elif str(out[c].dtype).startswith("float"):
            out[c] = pd.to_numeric(out[c], downcast="float")
    return out


class SessionStore:
    """Holds predictors (one per model dir) and loaded sessions."""

    def __init__(self, device: Optional[str] = None):
        self._device = device
        self._predictors: dict[str, InfiltrationPredictor] = {}
        self._sessions: dict[str, Session] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------ predictors

    def predictor(self, model_dir: Path | str) -> InfiltrationPredictor:
        key = str(model_dir)
        with self._lock:
            if key not in self._predictors:
                self._predictors[key] = InfiltrationPredictor(key, device=self._device)
            return self._predictors[key]

    def preload(self, specs) -> list:
        loaded = []
        for spec in specs:
            if spec.model_ready:
                try:
                    self.predictor(spec.model_dir)
                    loaded.append(spec.id)
                except Exception:  # a broken checkpoint must not stop the server
                    pass
        return loaded

    # -------------------------------------------------------------- sessions

    def get(self, session_id: str) -> Optional[Session]:
        return self._sessions.get(session_id)

    def create(self, spec: DatasetSpec, source: str) -> Session:
        predictor = self.predictor(spec.model_dir)
        session = Session(
            id=uuid.uuid4().hex[:12],
            dataset_id=spec.id,
            label=spec.label,
            source=source,
            window_seconds=int(predictor.cfg["window_seconds"]),
            window_unit=spec.window_unit,
            context_len=int(predictor.cfg["context_len"]),
            horizon_k=int(predictor.cfg["horizon_k"]),
        )
        self._sessions[session.id] = session
        return session

    def drop(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    def list_sessions(self) -> list:
        return list(self._sessions.values())

    # ------------------------------------------------------------- the work

    def load(self, session: Session, spec: DatasetSpec, csv_path: Path | str,
             nrows: Optional[int] = None) -> Session:
        """Read, window and score a capture. Blocking; run it in a threadpool."""
        p = session.progress
        try:
            p.stage, p.pct, p.message = "reading", 0.05, "Reading flow records"
            df = pd.read_csv(csv_path, nrows=nrows)

            missing = missing_columns(df)
            if missing:
                raise SchemaError(missing)

            session.n_rows = len(df)
            p.stage, p.pct, p.message = "windowing", 0.35, f"Windowing {len(df):,} flows"
            states, label_ids, timestamps, widx = build_state_sequence(
                df, session.window_seconds, with_index=True)

            p.stage, p.pct, p.message = "indexing", 0.65, "Indexing flows for inspection"
            session.display = _downcast_display(df)
            del df

            session.states = states
            session.label_ids = label_ids
            session.timestamps = timestamps
            session.window_index = widx

            predictor = self.predictor(spec.model_dir)
            session.states_norm = predictor.normalize(states).astype(np.float32)

            p.stage, p.pct, p.message = "scoring", 0.8, "Scoring the risk timeline"
            session.timeline = self._build_timeline(session, predictor)

            p.stage, p.pct, p.message, p.done = "ready", 1.0, "Ready", True
        except Exception as exc:
            p.error = str(exc)
            p.stage, p.done = "error", True
            raise
        return session

    @staticmethod
    def _build_timeline(session: Session, predictor: InfiltrationPredictor) -> dict:
        """One batched forward pass over every (downsampled) anchor in the capture."""
        L = session.context_len
        n = len(session.states_norm)
        if n < 1:
            return {"window_index": [], "t": [], "probability": [], "stage": [], "n_flows": []}

        first = min(L - 1, n - 1)
        anchors = np.arange(first, n)
        if len(anchors) > TIMELINE_MAX_POINTS:
            stride = int(np.ceil(len(anchors) / TIMELINE_MAX_POINTS))
            anchors = anchors[::stride]

        contexts = predictor.make_contexts(session.states_norm, anchors)
        scored = predictor.score_batch(contexts)

        counts = (session.window_index.ends - session.window_index.starts)[anchors]
        return {
            "window_index": anchors.astype(int).tolist(),
            "t": session.timestamps[anchors].astype(int).tolist(),
            "probability": [round(float(v), 5) for v in scored["infiltration_probability"]],
            "stage": [ID_TO_STAGE[int(i)] for i in scored["stage_id"]],
            "n_flows": counts.astype(int).tolist(),
            "truth": [ID_TO_STAGE[int(i)] for i in session.label_ids[anchors]]
                     if session.label_ids is not None else [],
        }
