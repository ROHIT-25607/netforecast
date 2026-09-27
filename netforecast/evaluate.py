"""
Benchmarks the world model against the baselines on the identical held-out
test split, and produces the evidence the "early warning" claim actually
rests on:

  1. Single-shot detection    -- precision/recall/F1/FPR, plus ROC-AUC and PR-AUC.
  2. Threshold sweep          -- the best-F1 operating point, not a hardcoded 0.5.
  3. Rollout vs horizon       -- K-step autoregressive forecast scored at each
                                 step against the truth at t+step. This is the
                                 differentiator, and it was previously unmeasured:
                                 every earlier number came from a single forward
                                 pass over observed context, i.e. exactly the
                                 plain sequence classifier the pitch claims to beat.
  4. Lead time                -- how many windows before the first labelled
                                 malicious flow the model crosses threshold.
  5. Ablation                 -- rollout vs. no-rollout on the same frozen weights.
  6. Latency                  -- featurization / forward / full rollout, ms.

Writes a Markdown report and the ROC/PR curves as PNGs (when matplotlib is
available). Never trains or modifies a checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Subset

from .dataset import NetworkStateSequenceDataset
from .features import build_state_sequence
from .predict import InfiltrationPredictor
from .world_model import WorldModel

BATCH = 512


# ----------------------------------------------------------------- metrics

def metrics_from_preds(y_true, y_pred) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fpr,
    }


def ranking_metrics(y_true, probs) -> dict:
    """Threshold-free quality. Guards the degenerate single-class case."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan")}
    return {"roc_auc": float(roc_auc_score(y_true, probs)),
            "pr_auc": float(average_precision_score(y_true, probs))}


def best_threshold(y_true, probs) -> tuple:
    """Operating point maximising F1, chosen on the given split."""
    best_f1, best_t = -1.0, 0.5
    for t in np.unique(np.round(np.asarray(probs), 3)):
        f1 = f1_score(y_true, (np.asarray(probs) >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, float(t)
    return best_t, best_f1


# -------------------------------------------------------------- inference

def collect_test(model, ds, test_idx, device) -> tuple:
    """Batched single-shot pass over the held-out anchors."""
    loader = DataLoader(Subset(ds, [int(i) for i in test_idx]), batch_size=BATCH, shuffle=False)
    probs, y = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["x"].to(device))
            probs.append(torch.sigmoid(out["infiltration_logit"]).cpu().numpy())
            y.append(b["infiltration"].numpy())
    return np.concatenate(y).astype(int), np.concatenate(probs)


def rollout_vs_horizon(predictor, states_norm, label_ids, anchors, k_max) -> list:
    """Score the K-step autoregressive rollout at every horizon step.

    At step s the model has fed its own predictions back in s-1 times, and we
    compare against whether any malicious window actually occurs in
    (t+1 .. t+s]. This measures the forecast, not the observation.
    """
    anchors = np.asarray(anchors)
    contexts = predictor.make_contexts(states_norm, anchors)
    out = predictor.rollout_batch(contexts, k=k_max)
    probs = out["infiltration_probability"]                 # (B, K)

    n = len(label_ids)
    rows = []
    for s in range(1, k_max + 1):
        keep, truth = [], []
        for i, t in enumerate(anchors):
            hi = t + s
            if hi >= n:
                continue
            keep.append(i)
            truth.append(int(np.max(label_ids[t + 1: hi + 1]) > 0))
        if not keep:
            continue
        p = probs[np.array(keep), s - 1]
        y = np.array(truth)
        m = metrics_from_preds(y, (p >= 0.5).astype(int))
        m.update(ranking_metrics(y, p))
        m.update({"step": s, "n": len(y), "positive_rate": float(y.mean())})
        rows.append(m)
    return rows


def lead_time(label_ids, anchors, probs, threshold: float, context_len: int) -> dict:
    """How early does the alarm fire, relative to the first malicious window
    of each attack episode?

    An episode is a maximal run of consecutive malicious windows. Only episodes
    whose onset falls inside the evaluated (test) anchor range are counted --
    scoring onsets the model was never asked about would inflate the result.
    For each one we walk back over the preceding benign windows and count how
    many consecutive ones the model was already above threshold for. A lead of
    0 means no warning before the first malicious flow appeared.
    """
    label_ids = np.asarray(label_ids)
    mal = label_ids > 0
    by_anchor = {int(a): float(p) for a, p in zip(anchors, probs)}
    if not by_anchor:
        return {"episodes": 0, "considered": 0, "warned": 0}

    episodes, i, n = [], 0, len(mal)
    while i < n:
        if mal[i]:
            j = i
            while j + 1 < n and mal[j + 1]:
                j += 1
            episodes.append(i)
            i = j + 1
        else:
            i += 1

    # Only onsets with a scored anchor immediately before them are assessable.
    # A day-aware split scatters test anchors across the whole capture, so a
    # plain min/max range would sweep in episodes sitting in training regions
    # and score them as "never warned".
    considered = [o for o in episodes if (o - 1) in by_anchor]

    leads = []
    for onset in considered:
        lead, t = 0, onset - 1
        while t >= max(context_len - 1, 0) and not mal[t]:
            p = by_anchor.get(t)
            if p is None or p < threshold:
                break
            lead += 1
            t -= 1
        leads.append(lead)

    if not leads:
        return {"episodes": len(episodes), "considered": 0, "warned": 0}
    arr = np.array(leads, dtype=float)
    warned = int((arr > 0).sum())
    return {
        "episodes": len(episodes),
        "considered": len(considered),
        "warned": warned,
        "warned_frac": warned / len(considered),
        "median": float(np.median(arr)),
        "median_warned": float(np.median(arr[arr > 0])) if warned else 0.0,
        "p90": float(np.percentile(arr, 90)),
        "mean": float(arr.mean()),
        "max": float(arr.max()),
    }


def latency_profile(predictor, df, states_norm, window_seconds, k, reps=20) -> dict:
    """Wall-clock cost of the pieces a live deployment pays per window."""
    L = predictor.cfg["context_len"]
    ctx = states_norm[-L:]

    sample = df.iloc[: min(len(df), 5000)]
    t0 = time.perf_counter()
    build_state_sequence(sample, window_seconds)
    feat_ms = (time.perf_counter() - t0) * 1000

    def timeit(fn, n):
        ts = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            ts.append((time.perf_counter() - t) * 1000)
        return float(np.median(ts)), float(np.percentile(ts, 99))

    fwd_med, fwd_p99 = timeit(lambda: predictor.score_batch(ctx[None, ...]), reps)
    roll_med, roll_p99 = timeit(lambda: predictor.rollout(ctx, k=k, explain=False), reps)
    exp_med, _ = timeit(lambda: predictor.rollout(ctx, k=k, explain=True), max(reps // 2, 4))

    return {"featurize_5k_flows_ms": feat_ms, "forward_ms": fwd_med, "forward_p99_ms": fwd_p99,
            "rollout_ms": roll_med, "rollout_p99_ms": roll_p99, "rollout_explain_ms": exp_med}


# ----------------------------------------------------------------- report

def _fmt(v, nd=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def build_report(ctx: dict) -> str:
    wm, bls = ctx["wm"], ctx["baselines"]
    L = ["# Benchmark: World Model vs Baselines\n",
         f"Dataset: `{ctx['data_path']}` · model: `{ctx['model_dir']}` · "
         f"held-out test anchors: **{ctx['n_test']}** "
         f"(positive rate {ctx['positive_rate']:.1%}).\n",
         "All numbers below come from frozen checkpoints; nothing here retrains the world model.\n",
         "\n## 1. Single-shot detection\n",
         "| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |",
         "|---|---|---|---|---|---|---|"]
    for name, m in bls:
        L.append(f"| {name} | {_fmt(m['precision'])} | {_fmt(m['recall'])} | {_fmt(m['f1'])} | "
                 f"{_fmt(m['false_positive_rate'])} | {_fmt(m.get('roc_auc'))} | {_fmt(m.get('pr_auc'))} |")
    L.append(f"| **LSTM World Model** | {_fmt(wm['precision'])} | {_fmt(wm['recall'])} | "
             f"**{_fmt(wm['f1'])}** | {_fmt(wm['false_positive_rate'])} | "
             f"{_fmt(wm.get('roc_auc'))} | {_fmt(wm.get('pr_auc'))} |")
    if bls:
        best = max(bls, key=lambda kv: kv[1]["f1"])
        L.append(f"\nF1 improvement over the strongest baseline ({best[0]}): "
                 f"**{wm['f1'] - best[1]['f1']:+.3f}**")

    t = ctx["threshold"]
    L += ["\n## 2. Operating point\n",
          f"Decision threshold is swept rather than fixed at 0.5. Best-F1 threshold on this split: "
          f"**{t['threshold']:.3f}** (F1 {t['f1']:.3f} vs {wm['f1']:.3f} at 0.5).",
          "\nNote this threshold is selected on the test split, so treat it as an upper bound; "
          "a deployment would tune it on validation."]

    L += ["\n## 3. Forecast quality vs horizon (the world-model claim)\n",
          "Step *s* is the model's **autoregressive rollout**: it has fed its own predicted state "
          "back in *s-1* times and has seen no new traffic. Scored against whether any malicious "
          "window actually occurs within *t+1 … t+s*.\n",
          "| Horizon step | n | Positive rate | Precision | Recall | F1 | FPR | ROC-AUC |",
          "|---|---|---|---|---|---|---|---|"]
    for r in ctx["horizon"]:
        L.append(f"| t+{r['step']} | {r['n']} | {r['positive_rate']:.1%} | {_fmt(r['precision'])} | "
                 f"{_fmt(r['recall'])} | {_fmt(r['f1'])} | {_fmt(r['false_positive_rate'])} | "
                 f"{_fmt(r.get('roc_auc'))} |")

    lt = ctx["lead"]
    unit, ws = ctx["window_unit"], ctx["window_seconds"]
    L += ["\n## 4. Lead time\n"]
    if lt.get("considered"):
        L += [f"Attack-episode onsets inside the evaluated test range: **{lt['considered']}** "
              f"(of {lt['episodes']} in the full capture). The model was already above the "
              f"{ctx['threshold']['threshold']:.2f} threshold before onset in **{lt['warned']}** "
              f"of them ({lt['warned_frac']:.0%}).\n",
              "| Statistic | Windows of warning | Equivalent |",
              "|---|---|---|",
              f"| Median (all onsets) | {lt['median']:.1f} | {lt['median'] * ws:.0f} {unit} |",
              f"| Median (onsets warned) | {lt['median_warned']:.1f} | {lt['median_warned'] * ws:.0f} {unit} |",
              f"| Mean | {lt['mean']:.1f} | {lt['mean'] * ws:.0f} {unit} |",
              f"| p90 | {lt['p90']:.1f} | {lt['p90'] * ws:.0f} {unit} |",
              f"| Max | {lt['max']:.1f} | {lt['max'] * ws:.0f} {unit} |",
              "\nLead is measured against the *labelled* onset, so this is warning raised before "
              "the first malicious flow was recorded -- not merely before an analyst noticed."]
    else:
        L.append("No attack-episode onsets fall inside the held-out split, so lead time is not "
                 "measurable on this dataset.")

    ab = ctx["ablation"]
    k = ab["k"]
    retention = 100 * ab["rollout_k"]["f1"] / max(ab["single"]["f1"], 1e-9)
    L += ["\n## 5. Ablation — what the rollout costs\n",
          f"Both rows predict the **same label** (any malicious window in *t+1 … t+{k}*), so they "
          f"are directly comparable. The difference is that the rollout has replaced {k - 1} of "
          f"its context updates with its *own* predicted states.\n",
          "| Configuration | F1 | ROC-AUC |", "|---|---|---|",
          f"| Single forward pass over observed context | {_fmt(ab['single']['f1'])} | {_fmt(ab['single'].get('roc_auc'))} |",
          f"| {k}-step autoregressive rollout | {_fmt(ab['rollout_k']['f1'])} | {_fmt(ab['rollout_k'].get('roc_auc'))} |",
          f"\nThe rollout retains **{retention:.0f}%** of single-shot F1 while running {k - 1} "
          f"steps on self-generated state. That retention is the evidence the learned transition "
          f"function is doing real work: a model that had only memorised a current-window mapping "
          f"would degrade sharply once fed its own output.",
          "\nThe per-step table in section 3 uses a *cumulative* label (malicious anywhere in "
          "*t+1 … t+s*), so its positive rate rises with s and early steps are not comparable "
          "with later ones or with the table above."]

    lat = ctx["latency"]
    L += ["\n## 6. Latency (CPU, single process)\n",
          "| Operation | Median | p99 |", "|---|---|---|",
          f"| Featurize 5k flows | {lat['featurize_5k_flows_ms']:.1f} ms | – |",
          f"| Single forward pass | {lat['forward_ms']:.2f} ms | {lat['forward_p99_ms']:.2f} ms |",
          f"| {ctx['horizon_k']}-step rollout | {lat['rollout_ms']:.2f} ms | {lat['rollout_p99_ms']:.2f} ms |",
          f"| Rollout + explainability | {lat['rollout_explain_ms']:.2f} ms | – |",
          "\nEverything runs offline on CPU — no GPU and no network call in the inference path."]

    if ctx.get("curves"):
        L.append(f"\n## Curves\n\n![ROC]({ctx['curves']['roc']})\n\n![PR]({ctx['curves']['pr']})")
    return "\n".join(L)


def save_curves(y, probs, out_dir, tag) -> Optional[dict]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import precision_recall_curve, roc_curve
    except Exception:
        return None
    if len(np.unique(y)) < 2:
        return None
    os.makedirs(out_dir, exist_ok=True)
    paths = {}
    fpr, tpr, _ = roc_curve(y, probs)
    for key, (xs, ys, xl, yl, title) in {
        "roc": (fpr, tpr, "False positive rate", "True positive rate", "ROC"),
        "pr": (*precision_recall_curve(y, probs)[1::-1], "Recall", "Precision", "Precision–Recall"),
    }.items():
        fig, ax = plt.subplots(figsize=(4.2, 3.4), dpi=140)
        ax.plot(xs, ys, color="#2a78d6", lw=2)
        if key == "roc":
            ax.plot([0, 1], [0, 1], color="#898781", lw=1, ls="--")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(f"{title} — {tag}")
        ax.grid(alpha=.25, lw=.6)
        fig.tight_layout()
        name = f"{key}_{tag}.png"
        fig.savefig(os.path.join(out_dir, name))
        plt.close(fig)
        paths[key] = name
    return paths


# -------------------------------------------------------------------- main

def main(model_dir="models", data_path="data/synthetic_flows.csv",
         report_path="reports/benchmark.md", window_unit="seconds"):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    norm = np.load(os.path.join(model_dir, "norm_stats.npz"))
    mean, std = norm["mean"], norm["std"]
    test_idx = np.load(os.path.join(model_dir, "split_indices.npz"))["test"]

    print(f"Loading {data_path} ...")
    df = pd.read_csv(data_path)
    states, label_ids, _, _ = build_state_sequence(df, cfg["window_seconds"])
    states_norm = ((states - mean) / std).astype(np.float32)
    print(f"{len(states)} windows, {len(test_idx)} test anchors")

    ds = NetworkStateSequenceDataset(states_norm, label_ids, cfg["context_len"], cfg["horizon_k"])
    device = torch.device("cpu")
    model = WorldModel(cfg["n_features"], cfg["n_stages"], cfg["hidden_dim"],
                       cfg["num_layers"]).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "world_model.pt"),
                                     map_location=device, weights_only=True))
    model.eval()

    # 1 & 2 -------------------------------------------------------------
    y_true, probs = collect_test(model, ds, test_idx, device)
    wm = metrics_from_preds(y_true, (probs >= 0.5).astype(int))
    wm.update(ranking_metrics(y_true, probs))
    thr, thr_f1 = best_threshold(y_true, probs)

    # baselines ----------------------------------------------------------
    baselines = []
    for fname, label in (("baseline_lr.joblib", "Logistic Regression"),
                         ("baseline_rf.joblib", "Random Forest")):
        path = os.path.join(model_dir, fname)
        bl_npz = os.path.join(model_dir, "baseline_test.npz")
        if not (os.path.exists(path) and os.path.exists(bl_npz)):
            continue
        clf = joblib.load(path)
        bl = np.load(bl_npz)
        X_test, y_bl = bl["X_test"], bl["y_test"]
        m = metrics_from_preds(y_bl, clf.predict(X_test))
        if hasattr(clf, "predict_proba"):
            m.update(ranking_metrics(y_bl, clf.predict_proba(X_test)[:, 1]))
        baselines.append((label, m))

    # 3 -------------------------------------------------------------------
    predictor = InfiltrationPredictor(model_dir, device="cpu")
    anchors = np.array([ds.valid_t[int(i)] for i in test_idx])
    print("Evaluating the K-step rollout ...")
    horizon = rollout_vs_horizon(predictor, states_norm, label_ids, anchors, cfg["horizon_k"])

    # 4 -------------------------------------------------------------------
    lead = lead_time(label_ids, anchors, probs, thr, cfg["context_len"])

    # 5 -------------------------------------------------------------------
    ablation = {"k": cfg["horizon_k"], "single": wm,
                "rollout_1": horizon[0] if horizon else wm,
                "rollout_k": horizon[-1] if horizon else wm}

    # 6 -------------------------------------------------------------------
    print("Profiling latency ...")
    latency = latency_profile(predictor, df, states_norm, cfg["window_seconds"], cfg["horizon_k"])

    out_dir = os.path.dirname(report_path) or "reports"
    tag = os.path.splitext(os.path.basename(report_path))[0]
    curves = save_curves(y_true, probs, out_dir, tag)

    report = build_report({
        "wm": wm, "baselines": baselines, "n_test": len(y_true),
        "positive_rate": float(y_true.mean()), "data_path": data_path, "model_dir": model_dir,
        "threshold": {"threshold": thr, "f1": thr_f1},
        "horizon": horizon, "lead": lead, "ablation": ablation, "latency": latency,
        "window_seconds": cfg["window_seconds"], "window_unit": window_unit,
        "horizon_k": cfg["horizon_k"], "curves": curves,
    })

    os.makedirs(out_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\n" + report)
    return wm, baselines


def cli():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--data", default="data/synthetic_flows.csv")
    ap.add_argument("--report", default="reports/benchmark.md")
    ap.add_argument("--window-unit", default="seconds", choices=["seconds", "flows"],
                    help="What the model's window_seconds actually counts. CIC-IDS2017 has no "
                         "usable clock, so its adapter windows by flow count.")
    a = ap.parse_args()
    main(a.model_dir, a.data, a.report, a.window_unit)


if __name__ == "__main__":
    cli()
