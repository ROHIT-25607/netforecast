"""
Benchmarks the world model's infiltration head against the logistic
regression baseline on the identical, chronologically held-out test split.
Reports F1, precision, recall and false-positive rate, and writes
reports/benchmark.md.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from dataset import NetworkStateSequenceDataset
from features import build_state_sequence
from world_model import WorldModel


def metrics_from_preds(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "false_positive_rate": fpr,
    }


def main(model_dir="models", data_path="data/synthetic_flows.csv", report_path="reports/benchmark.md"):
    with open(os.path.join(model_dir, "config.json")) as f:
        cfg = json.load(f)
    norm = np.load(os.path.join(model_dir, "norm_stats.npz"))
    mean, std = norm["mean"], norm["std"]
    split = np.load(os.path.join(model_dir, "split_indices.npz"))
    test_idx = split["test"]

    df = pd.read_csv(data_path)
    states, label_ids, _, _ = build_state_sequence(df, cfg["window_seconds"])
    states_norm = (states - mean) / std

    ds = NetworkStateSequenceDataset(states_norm, label_ids, cfg["context_len"], cfg["horizon_k"])

    device = torch.device("cpu")
    model = WorldModel(cfg["n_features"], cfg["n_stages"], cfg["hidden_dim"], cfg["num_layers"]).to(device)
    model.load_state_dict(torch.load(os.path.join(model_dir, "world_model.pt"), map_location=device))
    model.eval()

    y_true, y_pred_wm, probs_wm = [], [], []
    with torch.no_grad():
        for i in test_idx:
            item = ds[int(i)]
            x = item["x"].unsqueeze(0)
            out = model(x)
            p = torch.sigmoid(out["infiltration_logit"]).item()
            probs_wm.append(p)
            y_pred_wm.append(int(p >= 0.5))
            y_true.append(int(item["infiltration"].item()))

    wm_metrics = metrics_from_preds(y_true, y_pred_wm)

    # Baseline
    clf = joblib.load(os.path.join(model_dir, "baseline_lr.joblib"))
    bl = np.load(os.path.join(model_dir, "baseline_test.npz"))
    X_test, y_test_bl = bl["X_test"], bl["y_test"]
    y_pred_bl = clf.predict(X_test)
    bl_metrics = metrics_from_preds(y_test_bl, y_pred_bl)

    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    lines = [
        "# Benchmark: World Model vs Logistic Regression Baseline\n",
        f"Held-out, chronologically-future test set: {len(y_true)} windows.\n",
        "| Model | Precision | Recall | F1 | False Positive Rate |",
        "|---|---|---|---|---|",
        f"| Logistic Regression (baseline) | {bl_metrics['precision']:.3f} | {bl_metrics['recall']:.3f} | {bl_metrics['f1']:.3f} | {bl_metrics['false_positive_rate']:.3f} |",
        f"| LSTM World Model (attention, dynamics-trained) | {wm_metrics['precision']:.3f} | {wm_metrics['recall']:.3f} | {wm_metrics['f1']:.3f} | {wm_metrics['false_positive_rate']:.3f} |",
        "",
        f"F1 improvement over baseline: {wm_metrics['f1'] - bl_metrics['f1']:+.3f}",
    ]
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    return wm_metrics, bl_metrics


if __name__ == "__main__":
    main()
