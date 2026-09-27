"""Regenerate the feature golden fixtures.

`build_state_sequence` was rewritten from a per-window Python loop into
vectorized aggregation. The shipped checkpoints' normalization statistics were
fit on the loop's exact output, so any numerical drift would silently corrupt
every prediction while leaving the plumbing looking healthy.

This script recreates the **original loop** (using `compute_window_state` and
`window_label`, which are unchanged and kept as the reference implementation)
and stores its output. `tests/test_features.py` asserts the vectorized path
matches bit-for-bit.

    python tools/make_golden.py                       # the committed sample
    python tools/make_golden.py --csv data/synthetic_flows.csv --window-seconds 30
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from netforecast.features import compute_window_state, window_label  # noqa: E402
from netforecast.mitre_mapping import STAGE_TO_ID  # noqa: E402

FIXTURES = pathlib.Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def reference_build_state_sequence(df: pd.DataFrame, window_seconds: int):
    """The original loop implementation, preserved verbatim as the oracle."""
    df = df.copy()
    df["window_id"] = (df["timestamp"] // window_seconds).astype(int)
    t_min, t_max = df["window_id"].min(), df["window_id"].max()

    states, labels, timestamps, flow_indices = [], [], [], []
    grouped = {wid: g for wid, g in df.groupby("window_id")}
    for wid in range(int(t_min), int(t_max) + 1):
        win = grouped.get(wid, df.iloc[0:0])
        states.append(compute_window_state(win))
        labels.append(window_label(win))
        timestamps.append(wid * window_seconds)
        flow_indices.append(win.index.tolist())

    return (np.stack(states),
            np.array([STAGE_TO_ID[s] for s in labels], dtype=np.int64),
            np.array(timestamps),
            flow_indices)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default="data/sample_flows.csv")
    ap.add_argument("--window-seconds", type=int, default=30)
    ap.add_argument("--name", default=None, help="fixture basename (default: derived from --csv)")
    args = ap.parse_args()

    csv = pathlib.Path(args.csv)
    if not csv.exists():
        print(f"{csv} not found")
        return 1

    df = pd.read_csv(csv)
    t0 = time.perf_counter()
    states, label_ids, timestamps, flow_indices = reference_build_state_sequence(
        df, args.window_seconds)
    elapsed = time.perf_counter() - t0
    print(f"reference loop: {elapsed:.2f}s over {len(df):,} flows -> {states.shape}")

    lens = np.array([len(x) for x in flow_indices], dtype=np.int64)
    flat = (np.concatenate([np.array(x, dtype=np.int64) for x in flow_indices])
            if len(flow_indices) else np.array([], dtype=np.int64))

    stem = args.name or f"golden_{csv.stem.replace('_flows', '')}_w{args.window_seconds}"
    FIXTURES.mkdir(parents=True, exist_ok=True)
    out = FIXTURES / f"{stem}.npz"
    np.savez_compressed(out, states=states, label_ids=label_ids, timestamps=timestamps,
                        flow_lens=lens, flow_flat=flat, elapsed=np.array([elapsed]))
    print(f"wrote {out} ({out.stat().st_size / 1e3:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
