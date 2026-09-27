"""
Baselines for the benchmark. Each flattens the same L-step context window the
world model sees into a single vector and predicts the same K-step-ahead
infiltration label, so the comparison isolates *learning temporal dynamics*
from simply having the same features and horizon.

Two baselines are fitted:

* **Logistic regression** (class-balanced) -- the linear reference.
* **Random forest** -- a strong non-linear reference on identical inputs.

The random forest matters for credibility. On CIC-IDS2017 the logistic
regression is close to degenerate (it fires on ~83% of benign windows), and
"we beat a broken baseline" is not a result. A tuned tree ensemble on the same
330-dim vector is the comparison a reviewer will actually ask for.

Neither of these is the world model; running this script never touches
`world_model.pt`.
"""
from __future__ import annotations

import argparse
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .dataset import (
    CONTEXT_LEN,
    HORIZON_K,
    NetworkStateSequenceDataset,
    chronological_split,
    day_aware_split,
    resolve_data_path,
)
from .features import build_state_sequence

SEED = 42


def flatten_dataset(states_norm, label_ids, context_len, horizon_k, idx_filter=None):
    """Vectorized flatten of every context window into a (n, L*F) matrix."""
    ds = NetworkStateSequenceDataset(states_norm, label_ids, context_len, horizon_k)
    anchors = np.asarray(ds.valid_t if idx_filter is None
                         else [ds.valid_t[int(i)] for i in idx_filter])
    L, K = ds.L, ds.K
    offsets = np.arange(-L + 1, 1)
    rows = anchors[:, None] + offsets[None, :]              # (n, L)
    X = states_norm[rows].reshape(len(anchors), -1)

    fut = anchors[:, None] + np.arange(1, K + 1)[None, :]   # (n, K)
    y = (label_ids[fut].max(axis=1) > 0).astype(int)
    return X.astype(np.float32), y, ds


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic_flows.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--window-seconds", type=int, default=30)
    ap.add_argument("--no-rf", action="store_true", help="skip the random-forest baseline")
    ap.add_argument("--rf-trees", type=int, default=300)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    states, label_ids, _, _ = build_state_sequence(df, args.window_seconds)

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("day_boundaries"):
        with open(resolve_data_path(cfg["day_boundaries"])) as f:
            day_info = json.load(f)
        train_windows, _, _ = day_aware_split(day_info["row_counts_per_day"], args.window_seconds)
    else:
        tr_sl, _, _ = chronological_split(len(states))
        train_windows = list(range(tr_sl.start or 0, tr_sl.stop))

    # Normalize on train windows only -- same statistics the world model used.
    mean = states[train_windows].mean(axis=0)
    std = states[train_windows].std(axis=0) + 1e-6
    states_norm = ((states - mean) / std).astype(np.float32)

    split = np.load(os.path.join(args.model_dir, "split_indices.npz"))
    train_idx, test_idx = split["train"], split["test"]

    X_all, y_all, _ = flatten_dataset(states_norm, label_ids, CONTEXT_LEN, HORIZON_K)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]
    print(f"train {X_train.shape} · test {X_test.shape} · "
          f"positive rate train {y_train.mean():.1%} / test {y_test.mean():.1%}")

    os.makedirs(args.model_dir, exist_ok=True)

    lr = LogisticRegression(max_iter=2000, class_weight="balanced")
    lr.fit(X_train, y_train)
    joblib.dump(lr, os.path.join(args.model_dir, "baseline_lr.joblib"))
    print(f"logistic regression -> {args.model_dir}/baseline_lr.joblib")

    if not args.no_rf:
        rf = RandomForestClassifier(n_estimators=args.rf_trees, class_weight="balanced_subsample",
                                    min_samples_leaf=2, n_jobs=-1, random_state=SEED)
        rf.fit(X_train, y_train)
        joblib.dump(rf, os.path.join(args.model_dir, "baseline_rf.joblib"))
        print(f"random forest ({args.rf_trees} trees) -> {args.model_dir}/baseline_rf.joblib")

    np.savez(os.path.join(args.model_dir, "baseline_test.npz"), X_test=X_test, y_test=y_test)


if __name__ == "__main__":
    main()
