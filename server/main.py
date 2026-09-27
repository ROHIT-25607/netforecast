"""NetForecast API server.

    uvicorn server.main:app --reload

Serves the SOC dashboard at / and the JSON + WebSocket API under /api and /ws.
Interactive OpenAPI docs live at /docs.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from netforecast import __version__

from .config import DATASETS, DEFAULT_DATASET, STATIC_DIR, device_preference
from .ingest import IngestEngine
from .routers import api as api_router
from .routers import stream as stream_router
from .session import SchemaError, SessionStore

log = logging.getLogger("netforecast.server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    store = SessionStore(device=device_preference())
    ready = store.preload(DATASETS.values())

    app.state.store = store
    app.state.device = next(iter(store._predictors.values())).device if ready else "cpu"
    default_spec = DATASETS[DEFAULT_DATASET]
    app.state.ingest = IngestEngine(
        predictor_factory=lambda: store.predictor(default_spec.model_dir),
        # "seconds" datasets close a window on wall clock; CIC-IDS2017 has no
        # usable clock, so its windows close on a flow count instead.
        mode="time" if default_spec.window_unit == "seconds" else "count")

    if ready:
        log.info("Preloaded checkpoints: %s (device=%s)", ", ".join(ready), app.state.device)
    else:
        log.warning("No trained checkpoints found. Train one: python -m netforecast.train")
    yield


app = FastAPI(
    title="NetForecast API",
    version=__version__,
    lifespan=lifespan,
    description=(
        "An AI world model for network attack forecasting. Learns P(S_t+1 | S_t) over "
        "windowed network-traffic state and forecasts attacker progression K steps ahead, "
        "mapped to MITRE ATT&CK tactics. Runs fully offline."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # offline, on-prem tool; no credentialed cross-origin use
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router.router)
app.include_router(stream_router.router)


@app.exception_handler(SchemaError)
async def schema_error_handler(_request, exc: SchemaError):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc), "missing_columns": exc.missing},
    )


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard():
        return FileResponse(str(STATIC_DIR / "index.html"))


def run():
    """Console-script entry point: `netforecast-serve`."""
    import uvicorn
    uvicorn.run("server.main:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    run()
