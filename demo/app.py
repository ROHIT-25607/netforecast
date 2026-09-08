"""
Streamlit demo: AI Network Attack Forecasting.

Run with:  streamlit run demo/app.py
Fully offline: loads a locally-trained model checkpoint (models/) and a
CSV of flow records supplied by the user (or the bundled sample).
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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

@st.cache_data(show_spinner="Loading traffic + computing windowed network states (first run on a "
                             "large file can take a minute)...")
def load_and_window(path_or_buffer, window_seconds):
    df = pd.read_csv(path_or_buffer)
    states, label_ids, timestamps, flow_indices = build_state_sequence(df, window_seconds)
    return df, states, label_ids, timestamps, flow_indices


df, states, label_ids, timestamps, flow_indices = load_and_window(data_path, window_seconds)
states_norm = predictor.normalize(states)

st.subheader("1. Ingested traffic overview")
c1, c2, c3 = st.columns(3)
c1.metric("Flow records", f"{len(df):,}")
c2.metric("Time windows", f"{len(states):,}")
c3.metric("Window size", f"{window_seconds}s")

# Rolling infiltration-probability timeline: run rollout starting at every window
# (subsampled for speed on large files) using only the causal past as context.
st.subheader("2. Infiltration probability timeline")
@st.cache_data(show_spinner="Scanning traffic for the infiltration timeline...")
def compute_timeline(states_norm, timestamps, context_len):
    sample_stride = max(1, len(states_norm) // 300)
    rows = []
    for t in range(context_len - 1, len(states_norm), sample_stride):
        ctx = states_norm[t - context_len + 1: t + 1]
        # explain=False: skip the gradient/saliency pass here, it's only needed
        # once for the "current state" forecast below -- this loop runs ~300x.
        result = predictor.rollout(ctx, k=1, explain=False)
        rows.append({
            "window_index": t,
            "window_start_s": timestamps[t],
            "infiltration_probability": result["forecast"][0]["infiltration_probability"],
            "predicted_next_stage": result["forecast"][0]["predicted_stage"],
        })
    return pd.DataFrame(rows)


timeline_df = compute_timeline(states_norm, timestamps, context_len)
st.line_chart(timeline_df.set_index("window_start_s")["infiltration_probability"])

# --- Live replay animation -------------------------------------------------
STAGE_COLORS = {
    "Benign": "#2ecc71",
    "Reconnaissance": "#f1c40f",
    "Initial_Access": "#e67e22",
    "Lateral_Movement": "#e74c3c",
    "Command_And_Control": "#9b59b6",
    "Exfiltration": "#c0392b",
}
KILL_CHAIN = ["Reconnaissance", "Initial_Access", "Lateral_Movement", "Command_And_Control", "Exfiltration"]


def render_gauge(prob, stage):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob * 100,
        number={"suffix": "%"},
        title={"text": f"Infiltration risk — predicted stage: {stage.replace('_', ' ')}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": STAGE_COLORS.get(stage, "#2ecc71")},
            "steps": [
                {"range": [0, 33], "color": "rgba(46,204,113,0.15)"},
                {"range": [33, 66], "color": "rgba(241,196,15,0.15)"},
                {"range": [66, 100], "color": "rgba(231,76,60,0.15)"},
            ],
        },
    ))
    fig.update_layout(height=240, margin=dict(l=20, r=20, t=50, b=10))
    return fig


def render_stepper(current_stage):
    idx_current = KILL_CHAIN.index(current_stage) if current_stage in KILL_CHAIN else -1
    cols = st.columns(len(KILL_CHAIN))
    for i, (col, stage) in enumerate(zip(cols, KILL_CHAIN)):
        active = i <= idx_current
        color = STAGE_COLORS[stage] if active else "#3a3a3a"
        col.markdown(
            f"<div style='background:{color};padding:10px 4px;border-radius:8px;text-align:center;"
            f"color:white;font-size:12px;font-weight:600;'>{stage.replace('_', ' ')}</div>",
            unsafe_allow_html=True)


def render_growing_chart(times, probs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=times, y=probs, mode="lines", fill="tozeroy",
                              line=dict(color="#e74c3c", width=2)))
    fig.update_layout(height=220, margin=dict(l=20, r=20, t=10, b=10),
                       yaxis=dict(range=[0, 1], title="Infiltration probability"),
                       xaxis=dict(title="Time (s)"))
    return fig


st.subheader("🔴 Live Replay Simulation")
st.caption("Replays the ingested traffic window-by-window as if it were arriving live: watch the "
           "risk gauge and MITRE kill-chain stepper update as the model observes the attack unfold.")

rc1, rc2 = st.columns([1, 3])
with rc1:
    play = st.button("▶ Play Live Replay", width='stretch')
with rc2:
    speed = st.slider("Playback speed (windows/sec)", 1, 20, 8)

gauge_ph = st.empty()
stepper_ph = st.empty()
chart_ph = st.empty()
flows_ph = st.empty()

if play:
    running_probs, running_times = [], []
    for _, row in timeline_df.iterrows():
        stage = row["predicted_next_stage"]
        prob = row["infiltration_probability"]
        running_probs.append(prob)
        running_times.append(row["window_start_s"])

        gauge_ph.plotly_chart(render_gauge(prob, stage), width='stretch',
                               key=f"gauge_{row['window_index']}")
        with stepper_ph.container():
            render_stepper(stage)
        chart_ph.plotly_chart(render_growing_chart(running_times, running_probs),
                               width='stretch', key=f"chart_{row['window_index']}")

        t = int(row["window_index"])
        window_flows = flow_indices[t][:5] if t < len(flow_indices) else []
        with flows_ph.container():
            st.caption(f"Flows in this window (t={t}, showing up to 5):")
            if window_flows:
                st.dataframe(df.loc[window_flows], width='stretch', height=180)
            else:
                st.write("_No flows in this window._")

        time.sleep(1.0 / speed)
    st.success("Replay complete.")
else:
    gauge_ph.info("Click ▶ Play Live Replay to start the animated simulation.")

st.subheader("3. Current-state K-step forward simulation")
last_ctx = states_norm[-context_len:]
result = predictor.rollout(last_ctx, k=horizon_k)

risk = result["overall_infiltration_risk"]
risk_label = "🔴 HIGH" if risk > 0.66 else ("🟡 MEDIUM" if risk > 0.33 else "🟢 LOW")
st.metric("Overall infiltration risk (next K windows)", f"{risk:.2%}", risk_label)

forecast_df = pd.DataFrame(result["forecast"])
forecast_df["mitre_tactic"] = forecast_df["mitre_tactic"].apply(lambda d: f"{d['tactic_id']} {d['tactic_name']}")
st.dataframe(forecast_df[["step", "infiltration_probability", "predicted_stage", "mitre_tactic", "stage_confidence"]],
             width='stretch')

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
    st.dataframe(flagged, width='stretch')
else:
    st.info("No flows in the most recent window.")

st.subheader("MITRE ATT&CK stage reference")
st.table(pd.DataFrame(MITRE_TACTIC).T)
