"""Paths, dataset registry and tunables for the NetForecast API server."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
STATIC_DIR = Path(__file__).resolve().parent / "static"

# Columns kept resident per session for the flow-inspector panels. Holding
# these (as categoricals) instead of the full frame cuts a CIC-IDS2017
# session from ~1.0 GB to ~80 MB.
DISPLAY_COLUMNS = (
    "timestamp", "src_ip", "src_port", "dst_ip", "dst_port", "protocol",
    "duration", "tot_fwd_pkts", "tot_bwd_pkts",
    "totlen_fwd_bytes", "totlen_bwd_bytes", "label_stage",
)

TIMELINE_MAX_POINTS = 1200   # downsample the risk timeline to at most this many anchors
MAX_UPLOAD_BYTES = 512 * 1024 * 1024
RISK_HIGH, RISK_MEDIUM = 0.66, 0.33


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    label: str
    model_dir: Path
    sample_csv: Path
    #: CIC-IDS2017 has no usable wall-clock column, so `real_data_adapter`
    #: uses row order as the chronology proxy and `window_seconds` actually
    #: counts flows. Keep the unit explicit so the UI never says "seconds"
    #: about a number that is not seconds.
    window_unit: str
    description: str
    #: Whether the committed synthetic sample is a valid stand-in when this
    #: dataset's own capture is absent. Only true for datasets that share the
    #: sample's schema *and* windowing semantics -- scoring synthetic flows with
    #: a model trained on 200-flow windows and real-capture norm stats produces
    #: confident nonsense, which is worse than an honest "unavailable".
    fallback_to_sample: bool = True
    #: Shown when the capture is missing, so the gap is actionable.
    obtain: str = ""

    @property
    def available(self) -> bool:
        return self.model_ready and self.sample_csv.exists()

    @property
    def model_ready(self) -> bool:
        return all((self.model_dir / f).exists()
                   for f in ("world_model.pt", "config.json", "norm_stats.npz"))


DATASETS: dict[str, DatasetSpec] = {
    # Synthetic first: it is small, loads instantly and is the safe default
    # for a cold start or a live demo.
    "synthetic": DatasetSpec(
        id="synthetic",
        label="Synthetic campaign traffic",
        model_dir=ROOT / "models",
        sample_csv=ROOT / "data" / "synthetic_flows.csv",
        window_unit="seconds",
        description="48h of generated traffic with injected multi-stage kill-chain campaigns. "
                    "All 33 state features are populated.",
        fallback_to_sample=True,
        obtain="python -m netforecast.simulate_traffic --out data/synthetic_flows.csv",
    ),
    "cicids2017": DatasetSpec(
        id="cicids2017",
        label="CIC-IDS2017 (real capture)",
        model_dir=ROOT / "models_real",
        sample_csv=ROOT / "data" / "cicids2017_processed.csv",
        window_unit="flows",
        description="Real benign+attack capture, 2.45M flows. Packet-level features "
                    "(TTL, retransmission, fragmentation) are not recoverable from this "
                    "export and are zero-filled.",
        # 396 MB, so it cannot be committed. Never substitute the synthetic
        # sample here: this model windows by flow count and carries norm stats
        # from the real capture.
        fallback_to_sample=False,
        obtain="Download CIC-IDS2017 into data/raw/, then: "
               "python -m netforecast.real_data_adapter --raw-dir data/raw "
               "--out data/cicids2017_processed.csv --window-flow-count 200",
    ),
}

DEFAULT_DATASET = "synthetic"


def find_sample_csv(spec: DatasetSpec) -> Optional[Path]:
    """The dataset's own capture, or the committed sample when that is a valid
    stand-in for it. Returns None rather than substituting an incompatible
    capture -- a silently wrong answer is worse than a missing dataset."""
    if spec.sample_csv.exists():
        return spec.sample_csv
    if not spec.fallback_to_sample:
        return None
    fallback = ROOT / "data" / "sample_flows.csv"
    return fallback if fallback.exists() else None


def device_preference() -> Optional[str]:
    """Honour NETFORECAST_DEVICE, else let the predictor auto-select."""
    return os.environ.get("NETFORECAST_DEVICE") or None
