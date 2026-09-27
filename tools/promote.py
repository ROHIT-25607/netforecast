"""Promote a candidate checkpoint into a model directory, reversibly.

Copies the candidate's artifacts over the incumbent's after archiving the
incumbent under `<model_dir>/_previous/`, so a promotion can always be undone.
Refuses to run unless the candidate carries everything inference needs.

    python tools/promote.py --candidate C:/tmp/seed_42 --into models_real

Afterwards the baselines must be refit (the split changed) and the benchmark
regenerated:

    python -m netforecast.baseline --data <csv> --model-dir <dir> --window-seconds <w>
    python -m netforecast.evaluate --model-dir <dir> --data <csv> --report <report>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import shutil
import sys

REQUIRED = ("world_model.pt", "config.json", "norm_stats.npz", "split_indices.npz")
ALSO_COPY = ("history.json",)
# Baselines belong to the old split and must be refit, so they are not carried over.
STALE = ("baseline_lr.joblib", "baseline_rf.joblib", "baseline_test.npz")


def sha(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--into", required=True)
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    cand, dest = pathlib.Path(args.candidate), pathlib.Path(args.into)
    missing = [f for f in REQUIRED if not (cand / f).exists()]
    if missing:
        print(f"candidate is incomplete, missing: {', '.join(missing)}")
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    archive = dest / "_previous"
    archive.mkdir(exist_ok=True)
    print(f"Archiving current {dest}/ -> {archive}/")
    for f in REQUIRED + ALSO_COPY + STALE:
        src = dest / f
        if src.exists():
            shutil.copy2(src, archive / f)
            print(f"  kept {f}")

    print(f"\nPromoting {cand} -> {dest}")
    for f in REQUIRED + ALSO_COPY:
        if (cand / f).exists():
            shutil.copy2(cand / f, dest / f)
            print(f"  {f}  sha256={sha(dest / f)[:16]}")

    for f in STALE:
        p = dest / f
        if p.exists():
            p.unlink()
            print(f"  removed stale {f} (belongs to the previous split; refit required)")

    cfg = json.load(open(dest / "config.json"))
    print(f"\nPromoted. best_epoch={cfg.get('best_epoch')} "
          f"val_pr_auc={cfg.get('val_pr_auc')} temperature={cfg.get('temperature')}")
    print("Next: refit baselines, regenerate the benchmark, update the CI checksums.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
