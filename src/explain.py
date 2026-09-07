"""
SHAP-based explainability for the logistic-regression baseline, used
mainly in the benchmark report / demo to contrast with the world model's
built-in attention + saliency explanations (see predict.py).
"""
import numpy as np
import joblib
import shap

from dataset import CONTEXT_LEN
from features import STATE_FEATURE_NAMES


def flattened_feature_names(context_len=CONTEXT_LEN):
    names = []
    for t in range(context_len):
        for f in STATE_FEATURE_NAMES:
            names.append(f"t-{context_len - 1 - t}:{f}")
    return names


def explain_baseline(model_dir="models", n_background=100, n_explain=20):
    clf = joblib.load(f"{model_dir}/baseline_lr.joblib")
    bl = np.load(f"{model_dir}/baseline_test.npz")
    X_test = bl["X_test"]

    background = X_test[np.random.choice(len(X_test), min(n_background, len(X_test)), replace=False)]
    explainer = shap.LinearExplainer(clf, background)
    sample = X_test[:n_explain]
    shap_values = explainer.shap_values(sample)

    names = flattened_feature_names()
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(-mean_abs)[:15]
    ranking = [(names[i], float(mean_abs[i])) for i in order]
    return ranking


if __name__ == "__main__":
    for name, score in explain_baseline():
        print(f"{name:30s} {score:.4f}")
