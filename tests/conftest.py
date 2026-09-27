from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SAMPLE_CSV = ROOT / "data" / "sample_flows.csv"
SYNTHETIC_CSV = ROOT / "data" / "synthetic_flows.csv"
MODEL_DIR = ROOT / "models"


def _first_existing(*paths):
    for p in paths:
        if p.exists():
            return p
    return None


@pytest.fixture(scope="session")
def synthetic_csv() -> Path:
    """A capture to exercise the pipeline with.

    Prefers the full synthetic set when present, but a fresh clone only has the
    committed sample -- which is the case CI runs in.
    """
    p = _first_existing(SYNTHETIC_CSV, SAMPLE_CSV)
    if p is None:
        pytest.skip("no capture present; run `make sample`")
    return p


@pytest.fixture(scope="session")
def golden(synthetic_csv):
    """The golden fixture matching whichever capture is being used.

    Fixture and CSV must correspond, or the bit-exactness test compares the
    wrong things -- which is exactly what happened when a clean clone fell back
    from synthetic_flows.csv to sample_flows.csv.
    """
    name = {"synthetic_flows": "golden_synthetic_w30.npz",
            "sample_flows": "golden_sample_w30.npz"}.get(synthetic_csv.stem)
    if name is None or not (FIXTURES / name).exists():
        pytest.skip(f"no golden fixture for {synthetic_csv.name}; run tools/make_golden.py")
    return np.load(FIXTURES / name)


@pytest.fixture(scope="session")
def synthetic_df(synthetic_csv) -> pd.DataFrame:
    return pd.read_csv(synthetic_csv)


@pytest.fixture(scope="session")
def model_dir() -> Path:
    if not (MODEL_DIR / "world_model.pt").exists():
        pytest.skip("no trained checkpoint in models/")
    return MODEL_DIR


@pytest.fixture(scope="session")
def predictor(model_dir):
    from netforecast.predict import InfiltrationPredictor
    return InfiltrationPredictor(str(model_dir), device="cpu")


@pytest.fixture
def minimal_flows() -> pd.DataFrame:
    """A tiny schema-complete frame for edge-case tests."""
    n = 12
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "timestamp": np.arange(n, dtype=float) * 3,
        "src_ip": ["10.0.0.1"] * n,
        "dst_ip": ["10.0.0.2"] * (n - 3) + ["8.8.8.8"] * 3,
        "src_port": rng.integers(1024, 65535, n),
        "dst_port": [443, 80, 445] * (n // 3),
        "protocol": [6] * n,
        "duration": rng.random(n),
        "tot_fwd_pkts": rng.integers(1, 50, n),
        "tot_bwd_pkts": rng.integers(1, 50, n),
        "totlen_fwd_bytes": rng.integers(60, 5000, n),
        "totlen_bwd_bytes": rng.integers(60, 5000, n),
        "fwd_iat_mean": rng.random(n),
        "fwd_iat_std": rng.random(n),
        "fwd_iat_max": rng.random(n),
        "syn_flag_cnt": rng.integers(0, 2, n),
        "ack_flag_cnt": rng.integers(0, 2, n),
        "fin_flag_cnt": rng.integers(0, 2, n),
        "rst_flag_cnt": rng.integers(0, 2, n),
        "psh_flag_cnt": rng.integers(0, 2, n),
        "urg_flag_cnt": np.zeros(n, dtype=int),
        "init_win_bytes_fwd": rng.integers(0, 65535, n),
        "init_win_bytes_bwd": rng.integers(0, 65535, n),
        "ttl_mean": rng.integers(32, 128, n),
        "ttl_var": rng.random(n),
        "retransmission_cnt": np.zeros(n, dtype=int),
        "frag_flag": np.zeros(n, dtype=int),
        "label_stage": ["Benign"] * n,
    })
