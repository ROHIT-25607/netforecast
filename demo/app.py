"""
Streamlit demo: AI Network Attack Forecasting.

Run with:  streamlit run demo/app.py
Fully offline: loads a locally-trained model checkpoint (models/) and a
CSV of flow records supplied by the user (or the bundled sample).
"""
import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features import build_state_sequence, STATE_FEATURE_NAMES  # noqa: E402
from mitre_mapping import STAGES, MITRE_TACTIC  # noqa: E402
from predict import InfiltrationPredictor  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
MODEL_CHOICES = {
    "CIC-IDS2017 (real dataset)": {
        "model_dir": os.path.join(ROOT, "models_real"),
        "sample_csv": os.path.join(ROOT, "data", "cicids2017_processed.csv"),
    },
    "Synthetic demo dataset": {
        "model_dir": os.path.join(ROOT, "models"),
        "sample_csv": os.path.join(ROOT, "data", "synthetic_flows.csv"),
    },
}

st.set_page_config(page_title="Network Attack Forecasting World Model", layout="wide")
st.title("🛰️ AI World-Model for Network Attack Forecasting")
st.caption("Learns P(S_t+1 | S_t) over network traffic state and forecasts attacker progression "
           "K steps ahead, mapped to MITRE ATT&CK stages. Runs fully offline.")

with st.sidebar:
    dataset_choice = st.selectbox("Trained model / dataset", list(MODEL_CHOICES.keys()))
MODEL_DIR = MODEL_CHOICES[dataset_choice]["model_dir"]
SAMPLE_CSV = MODEL_CHOICES[dataset_choice]["sample_csv"]


@st.cache_resource
def load_predictor(model_dir):
    return InfiltrationPredictor(model_dir)


def check_model_ready(model_dir):
    required = ["world_model.pt", "config.json", "norm_stats.npz"]
    return all(os.path.exists(os.path.join(model_dir, r)) for r in required)


if not check_model_ready(MODEL_DIR):
    st.error(f"No trained model found in `{MODEL_DIR}`. Run:\n\n"
              "```\npython src/simulate_traffic.py\npython src/train.py\npython src/baseline.py\npython src/evaluate.py\n```\n"
              "or, for the real dataset:\n\n"
              "```\npython src/real_data_adapter.py\npython src/train.py --data data/cicids2017_processed.csv "
              "--window-seconds 200 --day-boundaries data/cicids2017_processed.day_boundaries.json --out-dir models_real\n```")
    st.stop()

predictor = load_predictor(MODEL_DIR)
window_seconds = predictor.cfg["window_seconds"]
context_len = predictor.cfg["context_len"]

with st.sidebar:
    st.header("Input")
    uploaded = st.file_uploader("Upload flow-record CSV (CICFlowMeter-style schema)", type=["csv"])
    horizon_k = st.slider("Forecast horizon K (windows)", 1, 15, predictor.cfg["horizon_k"])
    st.markdown("---")
    st.markdown("If no file is uploaded, the bundled synthetic sample is used.")

data_path = uploaded if uploaded is not None else (SAMPLE_CSV if os.path.exists(SAMPLE_CSV) else None)
if data_path is None:
    st.warning("No sample dataset found. Upload a CSV or run `python src/simulate_traffic.py` first.")
    st.stop()

df = pd.read_csv(data_path)
states, label_ids, timestamps, flow_indices = build_state_sequence(df, window_seconds)
states_norm = predictor.normalize(states)

st.subheader("1. Ingested traffic overview")
c1, c2, c3 = st.columns(3)
c1.metric("Flow records", f"{len(df):,}")
c2.metric("Time windows", f"{len(states):,}")
c3.metric("Window size", f"{window_seconds}s")

# Rolling infiltration-probability timeline: run rollout starting at every window
# (subsampled for speed on large files) using only the causal past as context.
st.subheader("2. Infiltration probability timeline")
sample_stride = max(1, len(states) // 300)
rows = []
for t in range(context_len - 1, len(states), sample_stride):
    ctx = states_norm[t - context_len + 1: t + 1]
    result = predictor.rollout(ctx, k=1)  # 1-step-ahead probability for the timeline view
    rows.append({
        "window_start_s": timestamps[t],
        "infiltration_probability": result["forecast"][0]["infiltration_probability"],
        "predicted_next_stage": result["forecast"][0]["predicted_stage"],
    })
timeline_df = pd.DataFrame(rows)
st.line_chart(timeline_df.set_index("window_start_s")["infiltration_probability"])

st.subheader("3. Current-state K-step forward simulation")
last_ctx = states_norm[-context_len:]
result = predictor.rollout(last_ctx, k=horizon_k)

risk = result["overall_infiltration_risk"]
risk_label = "🔴 HIGH" if risk > 0.66 else ("🟡 MEDIUM" if risk > 0.33 else "🟢 LOW")
st.metric("Overall infiltration risk (next K windows)", f"{risk:.2%}", risk_label)

forecast_df = pd.DataFrame(result["forecast"])
forecast_df["mitre_tactic"] = forecast_df["mitre_tactic"].apply(lambda d: f"{d['tactic_id']} {d['tactic_name']}")
st.dataframe(forecast_df[["step", "infiltration_probability", "predicted_stage", "mitre_tactic", "stage_confidence"]],
             use_container_width=True)

col_a, col_b = st.columns(2)
with col_a:
    st.subheader("4. Explainability — attention over recent windows")
    attn_df = pd.DataFrame({
        "context_step": [f"t-{context_len-1-i}" for i in range(context_len)],
        "attention_weight": result["attention_weights"],
    })
    st.bar_chart(attn_df.set_index("context_step"))
    st.caption("How much each of the last L observed windows influenced this prediction.")

with col_b:
    st.subheader("5. Explainability — top driving features")
    feat_df = pd.DataFrame(result["top_driving_features"], columns=["feature", "saliency"])
    st.bar_chart(feat_df.set_index("feature"))
    st.caption("Gradient x input saliency: which traffic features (flags, ports, timing) drove the risk score.")

st.subheader("6. Flagged flows (current window)")
current_window_flow_idx = flow_indices[-1]
if current_window_flow_idx:
    flagged = df.loc[current_window_flow_idx]
    st.dataframe(flagged, use_container_width=True)
else:
    st.info("No flows in the most recent window.")

st.subheader("MITRE ATT&CK stage reference")
st.table(pd.DataFrame(MITRE_TACTIC).T)
