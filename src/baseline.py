"""
Logistic regression baseline: flattens the same L-step context window used
by the world model into a single vector and predicts the same K-step-ahead
infiltration label. This isolates the benefit of *learning temporal
dynamics* (LSTM + attention) from simply having access to the same
features/horizon -- a fair, like-for-like benchmark comparison.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression

from dataset import (NetworkStateSequenceDataset, chronological_split, day_aware_split,
                      CONTEXT_LEN, HORIZON_K)
from features import build_state_sequence, N_FEATURES


def flatten_dataset(states_norm, label_ids, context_len, horizon_k, idx_filter=None):
    ds = NetworkStateSequenceDataset(states_norm, label_ids, context_len, horizon_k)
    X, y = [], []
    indices = range(len(ds)) if idx_filter is None else idx_filter
    for i in indices:
        item = ds[i]
        X.append(item["x"].numpy().flatten())
        y.append(item["infiltration"].item())
    return np.array(X), np.array(y), ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_flows.csv")
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--window-seconds", type=int, default=30)
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    states, label_ids, _, _ = build_state_sequence(df, args.window_seconds)

    with open(os.path.join(args.model_dir, "config.json")) as f:
        cfg = json.load(f)
    if cfg.get("day_boundaries"):
        with open(cfg["day_boundaries"]) as f:
            day_info = json.load(f)
        train_windows, _, _ = day_aware_split(day_info["row_counts_per_day"], args.window_seconds)
    else:
        tr_sl, _, _ = chronological_split(len(states))
        train_windows = list(range(tr_sl.start or 0, tr_sl.stop))
    mean = states[train_windows].mean(axis=0)
    std = states[train_windows].std(axis=0) + 1e-6
    states_norm = (states - mean) / std

    split = np.load(os.path.join(args.model_dir, "split_indices.npz"))
    train_idx, test_idx = split["train"], split["test"]

    X_all, y_all, ds = flatten_dataset(states_norm, label_ids, CONTEXT_LEN, HORIZON_K)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test, y_test = X_all[test_idx], y_all[test_idx]

    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X_train, y_train)

    os.makedirs(args.model_dir, exist_ok=True)
    joblib.dump(clf, os.path.join(args.model_dir, "baseline_lr.joblib"))
    np.savez(os.path.join(args.model_dir, "baseline_test.npz"), X_test=X_test, y_test=y_test)
    print("Baseline logistic regression trained and saved to models/baseline_lr.joblib")


if __name__ == "__main__":
    main()
