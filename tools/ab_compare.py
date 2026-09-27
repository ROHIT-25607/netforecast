"""Compare a candidate checkpoint against the incumbent on an identical split.

A retrain is only worth shipping if it is measurably better, so this scores both
checkpoints on the *same* test anchors, built from the candidate's split, and
prints a verdict.

Two deliberate choices keep the comparison conservative:

* Both models are evaluated on the corrected split. The incumbent was trained
  against a slightly misaligned one, so a couple of windows now in test may have
  been in its training set. That bias favours the incumbent -- so a candidate win
  is a safe conclusion, and a candidate loss is not necessarily a real one.
* Hyperparameters are selected on validation elsewhere; this script only reports.

    python tools/ab_compare.py --incumbent models_real --candidate models_real_v2 \
        --data data/cicids2017_processed.csv
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from netforecast.dataset import NetworkStateSequenceDataset  # noqa: E402
from netforecast.features import build_state_sequence  # noqa: E402
from netforecast.predict import InfiltrationPredictor  # noqa: E402


def load_states(data_path, window_seconds, model_dir):
    norm = np.load(f"{model_dir}/norm_stats.npz")
    df = pd.read_csv(data_path)
    states, label_ids, _, _ = build_state_sequence(df, window_seconds)
    return states, label_ids, ((states - norm["mean"]) / norm["std"]).astype(np.float32)


def evaluate(model_dir, data_path, anchors, label_ids, cfg_ref):
    """Score one checkpoint at a fixed set of anchor windows."""
    cfg = json.load(open(f"{model_dir}/config.json"))
    _, _, sn = load_states(data_path, cfg["window_seconds"], model_dir)
    p = InfiltrationPredictor(model_dir, device="cpu")

    K = cfg_ref["horizon_k"]
    n = len(label_ids)
    keep = [t for t in anchors if t + K < n]
    y = np.array([int(label_ids[t + 1:t + 1 + K].max() > 0) for t in keep])

    contexts = p.make_contexts(sn, keep)
    single = p.score_batch(contexts)["infiltration_probability"]
    rolled = p.rollout_batch(contexts, k=K)["infiltration_probability"][:, -1]

    def block(probs, tag):
        pred = (probs >= 0.5).astype(int)
        out = {
            f"{tag}_f1": f1_score(y, pred, zero_division=0),
            f"{tag}_precision": precision_score(y, pred, zero_division=0),
            f"{tag}_recall": recall_score(y, pred, zero_division=0),
        }
        if len(np.unique(y)) > 1:
            out[f"{tag}_roc_auc"] = roc_auc_score(y, probs)
            out[f"{tag}_pr_auc"] = average_precision_score(y, probs)
        out[f"{tag}_brier"] = brier_score_loss(y, probs)
        return out

    m = {"n": len(keep), "positive_rate": float(y.mean())}
    m.update(block(single, "single"))
    m.update(block(rolled, "rollout"))
    m["rollout_retention"] = (m["rollout_f1"] / m["single_f1"]) if m["single_f1"] else 0.0
    return m


HIGHER_IS_BETTER = {
    "single_f1": True, "single_roc_auc": True, "single_pr_auc": True,
    "rollout_f1": True, "rollout_roc_auc": True, "rollout_pr_auc": True,
    "rollout_retention": True, "single_brier": False, "rollout_brier": False,
}
# What the submission is actually judged on.
PRIMARY = ("single_f1", "single_pr_auc", "rollout_f1")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--split-from", default=None,
                    help="which model dir supplies the test anchors (default: candidate)")
    args = ap.parse_args()

    split_dir = args.split_from or args.candidate
    cfg = json.load(open(f"{split_dir}/config.json"))
    states, label_ids, sn = load_states(args.data, cfg["window_seconds"], split_dir)
    ds = NetworkStateSequenceDataset(sn, label_ids, cfg["context_len"], cfg["horizon_k"])
    test_idx = np.load(f"{split_dir}/split_indices.npz")["test"]
    anchors = [ds.valid_t[int(i)] for i in test_idx]
    print(f"Common test anchors from {split_dir}: {len(anchors)}\n")

    inc = evaluate(args.incumbent, args.data, anchors, label_ids, cfg)
    cand = evaluate(args.candidate, args.data, anchors, label_ids, cfg)

    print(f"{'metric':<22}{'incumbent':>12}{'candidate':>12}{'delta':>12}   ")
    print("-" * 62)
    for k in HIGHER_IS_BETTER:
        if k not in inc or k not in cand:
            continue
        d = cand[k] - inc[k]
        good = (d > 0) == HIGHER_IS_BETTER[k]
        mark = "" if abs(d) < 1e-9 else ("  better" if good else "  WORSE")
        print(f"{k:<22}{inc[k]:>12.4f}{cand[k]:>12.4f}{d:>+12.4f}{mark}")

    wins = sum(1 for k in PRIMARY
               if k in inc and (cand[k] - inc[k] > 0) == HIGHER_IS_BETTER[k]
               and abs(cand[k] - inc[k]) > 1e-4)
    losses = sum(1 for k in PRIMARY
                 if k in inc and (cand[k] - inc[k] < 0) == HIGHER_IS_BETTER[k]
                 and abs(cand[k] - inc[k]) > 1e-4)
    print(f"\nPrimary metrics {PRIMARY}: {wins} better, {losses} worse.")
    verdict = "PROMOTE" if wins > losses else "KEEP INCUMBENT"
    print(f"VERDICT: {verdict}")
    return 0 if verdict == "PROMOTE" else 2


if __name__ == "__main__":
    sys.exit(main())
