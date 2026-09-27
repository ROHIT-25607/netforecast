"""REST endpoints: datasets, sessions, timeline, forecast, explainability, flows."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, Request, UploadFile

from netforecast import __version__
from netforecast.mitre_mapping import MITRE_TACTIC, STAGES

from ..config import (
    DATASETS,
    DEFAULT_DATASET,
    MAX_UPLOAD_BYTES,
    RISK_HIGH,
    RISK_MEDIUM,
    DatasetSpec,
    find_sample_csv,
)
from ..schemas import (
    CreateSessionRequest,
    DatasetInfo,
    ExplainResponse,
    FlowsResponse,
    ForecastResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    LoadStatus,
    SessionSummary,
    TimelineResponse,
)
from ..session import Session

router = APIRouter(prefix="/api")


def _store(request: Request):
    return request.app.state.store


def _engine(request: Request):
    return request.app.state.ingest


def _session_or_404(request: Request, session_id: str) -> Session:
    s = _store(request).get(session_id)
    if s is None:
        raise HTTPException(404, f"unknown session '{session_id}'")
    return s


def _ready_or_409(s: Session) -> Session:
    if s.progress.error:
        raise HTTPException(409, f"session failed to load: {s.progress.error}")
    if not s.progress.done or s.states_norm is None:
        raise HTTPException(409, f"session is still loading ({s.progress.stage}, "
                                 f"{s.progress.pct:.0%})")
    return s


def _risk_band(risk: float) -> str:
    return "HIGH" if risk > RISK_HIGH else ("MEDIUM" if risk > RISK_MEDIUM else "LOW")


def _summary(s: Session) -> SessionSummary:
    return SessionSummary(
        session_id=s.id, dataset_id=s.dataset_id, label=s.label, source=s.source,
        n_rows=s.n_rows, n_windows=s.n_windows, window_seconds=s.window_seconds,
        window_unit=s.window_unit, context_len=s.context_len, horizon_k=s.horizon_k,
        ready=s.progress.done and s.progress.error is None,
    )


# --------------------------------------------------------------------- meta

@router.get("/health", response_model=HealthResponse)
def health(request: Request):
    store = _store(request)
    return HealthResponse(
        status="ok",
        version=__version__,
        device=str(request.app.state.device),
        datasets_ready=[d.id for d in DATASETS.values() if d.model_ready],
        sessions=len(store.list_sessions()),
    )


@router.get("/datasets", response_model=list[DatasetInfo])
def datasets(request: Request):
    store = _store(request)
    out = []
    for spec in DATASETS.values():
        cfg = {}
        if spec.model_ready:
            try:
                cfg = store.predictor(spec.model_dir).cfg
            except Exception:
                cfg = {}
        out.append(DatasetInfo(
            id=spec.id, label=spec.label, description=spec.description,
            available=spec.model_ready and find_sample_csv(spec) is not None,
            model_ready=spec.model_ready,
            sample_present=find_sample_csv(spec) is not None,
            window_unit=spec.window_unit,
            window_seconds=cfg.get("window_seconds"),
            context_len=cfg.get("context_len"),
            horizon_k=cfg.get("horizon_k"),
        ))
    return out


@router.get("/mitre")
def mitre():
    return {"stages": STAGES, "tactics": MITRE_TACTIC}


# ----------------------------------------------------------------- sessions

def _spec_or_400(dataset_id: str) -> DatasetSpec:
    spec = DATASETS.get(dataset_id)
    if spec is None:
        raise HTTPException(400, f"unknown dataset '{dataset_id}'; "
                                 f"choose one of {list(DATASETS)}")
    if not spec.model_ready:
        raise HTTPException(
            409,
            f"no trained checkpoint in {spec.model_dir.name}/. Train one first: "
            f"python -m netforecast.train --out-dir {spec.model_dir.name}")
    return spec


def _load_task(store, session: Session, spec: DatasetSpec, path: Path,
               nrows: Optional[int], cleanup: Optional[Path] = None):
    try:
        store.load(session, spec, path, nrows=nrows)
    except Exception:
        pass  # the failure is recorded on session.progress and surfaced by /status
    finally:
        if cleanup is not None:
            cleanup.unlink(missing_ok=True)


@router.post("/session", response_model=SessionSummary, status_code=202)
def create_session(request: Request, body: CreateSessionRequest, bg: BackgroundTasks):
    """Start loading a bundled dataset. Poll /api/session/{id}/status for progress."""
    spec = _spec_or_400(body.dataset or DEFAULT_DATASET)
    csv_path = find_sample_csv(spec)
    if csv_path is None:
        raise HTTPException(
            409,
            f"no sample capture for '{spec.id}'. Generate one with "
            f"python -m netforecast.simulate_traffic, or upload a CSV.")
    store = _store(request)
    session = store.create(spec, source=csv_path.name)
    bg.add_task(_load_task, store, session, spec, csv_path, body.nrows)
    return _summary(session)


@router.post("/session/upload", response_model=SessionSummary, status_code=202)
async def upload_session(request: Request, bg: BackgroundTasks,
                         file: UploadFile = File(...),
                         dataset: str = Query(DEFAULT_DATASET,
                                              description="Which trained model scores this capture"),
                         nrows: Optional[int] = Query(None, ge=1)):
    """Score an uploaded CSV with the chosen model."""
    spec = _spec_or_400(dataset)
    if not (file.filename or "").lower().endswith(".csv"):
        raise HTTPException(415, "please upload a .csv of flow records")

    tmp = Path(tempfile.gettempdir()) / f"netforecast_upload_{file.filename}"
    size = 0
    with tmp.open("wb") as fh:
        while chunk := await file.read(1 << 20):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                tmp.unlink(missing_ok=True)
                raise HTTPException(413, f"upload exceeds {MAX_UPLOAD_BYTES // (1 << 20)} MB")
            fh.write(chunk)

    store = _store(request)
    session = store.create(spec, source=file.filename or "upload.csv")
    bg.add_task(_load_task, store, session, spec, tmp, nrows, tmp)
    return _summary(session)


@router.get("/session/{session_id}/status", response_model=LoadStatus)
def session_status(request: Request, session_id: str):
    s = _session_or_404(request, session_id)
    p = s.progress
    return LoadStatus(session_id=s.id, stage=p.stage, pct=p.pct, message=p.message,
                      done=p.done, error=p.error)


@router.get("/session/{session_id}", response_model=SessionSummary)
def session_summary(request: Request, session_id: str):
    return _summary(_session_or_404(request, session_id))


@router.delete("/session/{session_id}")
def delete_session(request: Request, session_id: str):
    if not _store(request).drop(session_id):
        raise HTTPException(404, f"unknown session '{session_id}'")
    return {"deleted": session_id}


# ---------------------------------------------------------------- analytics

@router.get("/session/{session_id}/timeline", response_model=TimelineResponse)
def timeline(request: Request, session_id: str):
    s = _ready_or_409(_session_or_404(request, session_id))
    return TimelineResponse(session_id=s.id, window_unit=s.window_unit, **s.timeline)


@router.get("/session/{session_id}/forecast", response_model=ForecastResponse)
def forecast(request: Request, session_id: str,
             k: Optional[int] = Query(None, ge=1, le=30),
             window: Optional[int] = Query(None, ge=0,
                                           description="Anchor window; defaults to the latest")):
    s = _ready_or_409(_session_or_404(request, session_id))
    spec = DATASETS[s.dataset_id]
    predictor = _store(request).predictor(spec.model_dir)

    anchor = s.n_windows - 1 if window is None else min(window, s.n_windows - 1)
    lo = max(0, anchor - s.context_len + 1)
    result = predictor.rollout(s.states_norm[lo:anchor + 1], k=k or s.horizon_k, explain=False)
    risk = result["overall_infiltration_risk"]
    return ForecastResponse(
        session_id=s.id, anchor_window=anchor, horizon_k=k or s.horizon_k,
        overall_infiltration_risk=risk, risk_band=_risk_band(risk),
        forecast=result["forecast"],
    )


@router.get("/session/{session_id}/explain", response_model=ExplainResponse)
def explain(request: Request, session_id: str,
            window: Optional[int] = Query(None, ge=0)):
    s = _ready_or_409(_session_or_404(request, session_id))
    spec = DATASETS[s.dataset_id]
    predictor = _store(request).predictor(spec.model_dir)

    anchor = s.n_windows - 1 if window is None else min(window, s.n_windows - 1)
    lo = max(0, anchor - s.context_len + 1)
    result = predictor.rollout(s.states_norm[lo:anchor + 1], k=1, explain=True)

    weights = result["attention_weights"] or []
    attention = [{"context_step": f"t-{len(weights) - 1 - i}", "weight": round(float(w), 6)}
                 for i, w in enumerate(weights)]
    top = [{"feature": n, "saliency": round(float(v), 6)}
           for n, v in result["top_driving_features"]]
    return ExplainResponse(session_id=s.id, anchor_window=anchor,
                           context_len=s.context_len, attention=attention, top_features=top)


@router.get("/session/{session_id}/flows", response_model=FlowsResponse)
def flows(request: Request, session_id: str,
          window: Optional[int] = Query(None, ge=0),
          limit: int = Query(50, ge=1, le=500)):
    s = _ready_or_409(_session_or_404(request, session_id))
    w = s.n_windows - 1 if window is None else min(window, s.n_windows - 1)
    rows = s.flows_for_window(w, limit=limit)
    cols = list(s.display.columns) if s.display is not None else []
    return FlowsResponse(session_id=s.id, window=w, total_in_window=s.window_flow_count(w),
                         returned=len(rows), columns=cols, rows=rows)


@router.get("/session/{session_id}/state")
def window_state(request: Request, session_id: str, window: Optional[int] = Query(None, ge=0)):
    """The raw 33-dim state vector for a window -- useful for debugging and for
    showing judges that the model consumes an interpretable feature vector."""
    from netforecast.features import STATE_FEATURE_NAMES
    s = _ready_or_409(_session_or_404(request, session_id))
    w = s.n_windows - 1 if window is None else min(window, s.n_windows - 1)
    raw = s.states[w]
    norm = s.states_norm[w]
    return {
        "session_id": s.id, "window": w,
        "features": [{"name": n, "raw": float(raw[i]), "normalized": float(norm[i])}
                     for i, n in enumerate(STATE_FEATURE_NAMES)],
    }


# ------------------------------------------------------------- live ingest

@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, body: IngestRequest):
    """Push live flow records in. Completed windows are scored and broadcast
    to any dashboard subscribed to this source over /ws/live/{source_id}."""
    engine = _engine(request)
    try:
        result = await engine.add_flows(body.source_id, body.flows,
                                        body.window_size, body.mode)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    return IngestResponse(source_id=body.source_id, accepted=result["accepted"],
                          buffered=result.get("buffered", 0), windows=result["windows"])


@router.post("/ingest/{source_id}/flush", response_model=IngestResponse)
async def ingest_flush(request: Request, source_id: str):
    engine = _engine(request)
    result = await engine.flush(source_id)
    src = engine.sources.get(source_id)
    return IngestResponse(source_id=source_id, accepted=0,
                          buffered=len(src.buffer) if src else 0,
                          windows=result["windows"])


@router.get("/ingest/{source_id}")
async def ingest_status(request: Request, source_id: str):
    src = _engine(request).sources.get(source_id)
    if src is None:
        raise HTTPException(404, f"no live source '{source_id}'")
    return src.snapshot()


@router.delete("/ingest/{source_id}")
async def ingest_reset(request: Request, source_id: str):
    if not _engine(request).reset(source_id):
        raise HTTPException(404, f"no live source '{source_id}'")
    return {"reset": source_id}
