"""Pydantic response/request models for the NetForecast API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    id: str
    label: str
    description: str
    available: bool
    model_ready: bool
    sample_present: bool
    window_unit: str
    window_seconds: Optional[int] = None
    context_len: Optional[int] = None
    horizon_k: Optional[int] = None


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str
    datasets_ready: list[str]
    sessions: int


class CreateSessionRequest(BaseModel):
    dataset: str = Field(..., description="Dataset id from GET /api/datasets")
    nrows: Optional[int] = Field(None, ge=1, description="Cap rows read, for a fast preview")


class SessionSummary(BaseModel):
    session_id: str
    dataset_id: str
    label: str
    source: str
    n_rows: int
    n_windows: int
    window_seconds: int
    window_unit: str
    context_len: int
    horizon_k: int
    ready: bool


class LoadStatus(BaseModel):
    session_id: str
    stage: str
    pct: float
    message: str
    done: bool
    error: Optional[str] = None


class TimelineResponse(BaseModel):
    session_id: str
    window_index: list[int]
    t: list[int]
    probability: list[float]
    stage: list[str]
    n_flows: list[int]
    truth: list[str] = []
    window_unit: str


class ForecastStep(BaseModel):
    step: int
    infiltration_probability: float
    predicted_stage: str
    mitre_tactic: dict
    stage_confidence: float


class ForecastResponse(BaseModel):
    session_id: str
    anchor_window: int
    horizon_k: int
    overall_infiltration_risk: float
    risk_band: str
    forecast: list[ForecastStep]


class ExplainResponse(BaseModel):
    session_id: str
    anchor_window: int
    context_len: int
    attention: list[dict]
    top_features: list[dict]


class FlowsResponse(BaseModel):
    session_id: str
    window: int
    total_in_window: int
    returned: int
    columns: list[str]
    rows: list[dict]


class IngestRequest(BaseModel):
    source_id: str = Field("default", description="Logical sensor / tap identifier")
    flows: list[dict[str, Any]] = Field(..., description="Flow records in the project schema")
    window_size: Optional[int] = Field(None, ge=1,
                                       description="Override the window size (seconds, or flows)")
    mode: Optional[str] = Field(None, pattern="^(time|count)$",
                                description="How a window closes: 'time' (seconds of wall clock) "
                                            "or 'count' (consecutive flows). Defaults to whatever "
                                            "the active model was trained with.")


class IngestResponse(BaseModel):
    source_id: str
    accepted: int
    buffered: int
    windows: list[dict]
