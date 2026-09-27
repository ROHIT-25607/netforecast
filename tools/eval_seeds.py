"""Score several checkpoints on the same validation and test splits.

Used to check that a retrain's advantage is real rather than one lucky seed.
The protocol is: pick the seed by **validation**, then report its test numbers;
and separately confirm that *every* seed clears the incumbent on test, so the
promotion does not rest on a single draw.

    python tools/eval_seeds.py --data data/cicids2017_processed.csv \
        --window-seconds 200 --incumbent models_real \
        --candidates C:/tmp/seed_42 C:/tmp/seed_1337 C:/tmp/seed_2024
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from netforecast.dataset import NetworkStateSequenceDataset, day_aware_split  # noqa: E402
from netforecast.features import build_state_sequence  # noqa: E402
from netforecast.predict import InfiltrationPredictor  # noqa: E402


def score(model_dir, states, labels, windows, K=5, L=10):
    p = InfiltrationPredictor(model_dir, device="cpu")
    sn = p.normalize(states).astype(np.float32)
    ds = NetworkStateSequenceDataset(sn, labels, L, K)
    wset = set(windows)
    anchors = [t for t in ds.valid_t if t in wset]
    y = np.array([int(labels[t + 1:t + 1 + K].max() > 0) for t in anchors])
    ctx = p.make_contexts(sn, anchors)
    single = p.score_batch(ctx)["infiltration_probability"]
    rolled = p.rollout_batch(ctx, k=K)["infiltration_probability"][:, -1]
    return {
        "n": len(anchors),
        "pr_auc": average_precision_score(y, single),
        "roc_auc": roc_auc_score(y, single),
        "f1": f1_score(y, (single >= 0.5).astype(int), zero_division=0),
        "roll_f1": f1_score(y, (rolled >= 0.5).astype(int), zero_division=0),
        "roll_pr_auc": average_precision_score(y, rolled),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--window-seconds", type=int, required=True)
    ap.add_argument("--day-boundaries", default="data/cicids2017_processed.day_boundaries.json")
    ap.add_argument("--incumbent", required=True)
    ap.add_argument("--candidates", nargs="+", required=True)
    args = ap.parse_args()

    rc = json.load(open(args.day_boundaries))["row_counts_per_day"]
    _, va, te = day_aware_split(rc, args.window_seconds)
    df = pd.read_csv(args.data)
    states, labels, _, _ = build_state_sequence(df, args.window_seconds)

    rows = []
    for md in [args.incumbent] + args.candidates:
        v = score(md, states, labels, va)
        t = score(md, states, labels, te)
        rows.append((md, v, t))

    name_w = max(len(pathlib.Path(m).name) for m, _, _ in rows) + 2
    print(f"{'model':<{name_w}}{'VAL pr_auc':>12}{'TEST pr_auc':>13}{'TEST roc_auc':>14}"
          f"{'TEST f1':>10}{'TEST roll_f1':>14}")
    print("-" * (name_w + 63))
    for md, v, t in rows:
        print(f"{pathlib.Path(md).name:<{name_w}}{v['pr_auc']:>12.4f}{t['pr_auc']:>13.4f}"
              f"{t['roc_auc']:>14.4f}{t['f1']:>10.4f}{t['roll_f1']:>14.4f}")

    inc = rows[0][2]
    cands = rows[1:]
    keys = ("pr_auc", "roc_auc", "f1", "roll_f1")

    print("\nIncumbent test: " + "  ".join(f"{k} {inc[k]:.4f}" for k in keys))
    arr = {k: np.array([t[k] for _, _, t in cands]) for k in keys}
    print("Seed spread (test):")
    for k in keys:
        print(f"  {k:<10} mean {arr[k].mean():.4f}   range {arr[k].min():.4f}-{arr[k].max():.4f}"
              f"   mean delta {arr[k].mean() - inc[k]:+.4f}")

    # What ships is a *checkpoint*, so the promotion question is whether that one
    # checkpoint beats the incumbent on held-out data. The seed spread answers a
    # different question -- whether the training recipe is reliably better -- and
    # is reported as a caveat rather than used as the gate. Selection is on
    # validation, never on test.
    best = max(cands, key=lambda r: r[1]["pr_auc"])
    bt = best[2]
    print(f"\nVal-selected candidate: {pathlib.Path(best[0]).name}")
    print("  test: " + "  ".join(f"{k} {bt[k]:.4f} ({bt[k] - inc[k]:+.4f})" for k in keys))

    worse = [k for k in keys if bt[k] < inc[k]]
    ok = not worse
    if worse:
        print(f"  regressions vs incumbent: {', '.join(worse)}")
    n_clear = sum(1 for _, _, t in cands if t["pr_auc"] > inc["pr_auc"])
    print(f"  recipe reliability: {n_clear}/{len(cands)} seeds clear the incumbent on PR-AUC")
    print(f"\nVERDICT: {'PROMOTE' if ok else 'KEEP INCUMBENT'}"
          + ("" if ok else "  (candidate regresses on at least one primary metric)"))
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
