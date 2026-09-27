"""WebSocket endpoints: replay of a loaded capture, and live-ingest subscription.

The replay pushes one scored window per tick from a precomputed timeline, so
play / pause / seek / speed are instant and never recompute anything. The
crucial difference from the old Streamlit loop is that the sleep happens in
the event loop, not on the thread rendering the page -- the rest of the
dashboard stays fully interactive while a replay runs.
"""
from __future__ import annotations

import asyncio
import contextlib

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..config import RISK_HIGH, RISK_MEDIUM

router = APIRouter()

MIN_SPEED, MAX_SPEED = 0.25, 60.0
FLOW_PREVIEW_ROWS = 6


def _band(p: float) -> str:
    return "HIGH" if p > RISK_HIGH else ("MEDIUM" if p > RISK_MEDIUM else "LOW")


class ReplayState:
    def __init__(self, n: int, speed: float = 8.0):
        self.cursor = 0
        self.n = n
        self.speed = speed
        self.playing = False


@router.websocket("/ws/replay/{session_id}")
async def replay(ws: WebSocket, session_id: str):
    await ws.accept()
    store = ws.app.state.store
    session = store.get(session_id)

    if session is None:
        await ws.send_json({"type": "error", "message": f"unknown session '{session_id}'"})
        await ws.close()
        return
    if not session.progress.done or session.timeline is None:
        await ws.send_json({"type": "error",
                            "message": f"session still loading ({session.progress.stage})"})
        await ws.close()
        return

    tl = session.timeline
    state = ReplayState(n=len(tl["window_index"]))

    await ws.send_json({
        "type": "ready",
        "session_id": session_id,
        "n_frames": state.n,
        "window_unit": session.window_unit,
        "window_seconds": session.window_seconds,
        "speed": state.speed,
    })

    async def emit(i: int) -> None:
        i = max(0, min(i, state.n - 1))
        w = tl["window_index"][i]
        prob = tl["probability"][i]
        await ws.send_json({
            "type": "frame",
            "frame": i,
            "window_index": w,
            "t": tl["t"][i],
            "probability": prob,
            "stage": tl["stage"][i],
            "band": _band(prob),
            "n_flows": tl["n_flows"][i],
            "truth": tl["truth"][i] if tl.get("truth") else None,
            "flows": session.flows_for_window(w, limit=FLOW_PREVIEW_ROWS),
            "progress": (i + 1) / max(state.n, 1),
        })

    async def player() -> None:
        try:
            while True:
                if not state.playing:
                    await asyncio.sleep(0.05)
                    continue
                if state.cursor >= state.n:
                    state.playing = False
                    await ws.send_json({"type": "complete", "frames": state.n})
                    continue
                await emit(state.cursor)
                state.cursor += 1
                await asyncio.sleep(1.0 / state.speed)
        except (WebSocketDisconnect, RuntimeError):
            pass

    task = asyncio.create_task(player())
    try:
        while True:
            msg = await ws.receive_json()
            action = msg.get("action")
            if action == "play":
                if state.cursor >= state.n:
                    state.cursor = 0
                state.playing = True
            elif action == "pause":
                state.playing = False
            elif action == "speed":
                state.speed = max(MIN_SPEED, min(MAX_SPEED, float(msg.get("value", 8))))
                await ws.send_json({"type": "speed", "value": state.speed})
            elif action == "seek":
                state.cursor = max(0, min(int(msg.get("frame", 0)), state.n - 1))
                await emit(state.cursor)
            elif action == "restart":
                state.cursor = 0
                state.playing = True
            elif action == "frame":
                await emit(int(msg.get("frame", state.cursor)))
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


@router.websocket("/ws/live/{source_id}")
async def live(ws: WebSocket, source_id: str):
    """Subscribe to windows scored from flows arriving via POST /api/ingest."""
    await ws.accept()
    engine = ws.app.state.ingest
    src = engine.subscribe(source_id, ws)
    await ws.send_json({"type": "ready", "source_id": source_id, **src.snapshot()})
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("action") == "snapshot":
                await ws.send_json({"type": "snapshot", **src.snapshot()})
            elif msg.get("action") == "reset":
                engine.reset(source_id)
                src = engine.subscribe(source_id, ws)
                await ws.send_json({"type": "snapshot", **src.snapshot()})
    except WebSocketDisconnect:
        pass
    finally:
        engine.unsubscribe(source_id, ws)
