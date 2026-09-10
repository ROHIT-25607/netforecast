# Project Presentation

**PPT:** [Open Final Presentation](./NetForecast_SIH2026_Presentation.pptx)

**Google Slides:** [View / Present](https://docs.google.com/presentation/d/19jCtfTlM789KAU1H4Gnc4WW3YucuD-b_/edit?usp=sharing&ouid=110188210868834090212&rtpof=true&sd=true)

9 slides: Title → Reactive vs Proactive → State Representation →
World Model Architecture → K-Step Rollout & Kill-Chain Mapping →
Explainability → Results & Impact → Key Features & Tech Stack →
Future Scope.

---

## Slide-by-slide notes (for narration)

## Slide 1 — Title
NetForecast — AI World Model for Network Attack Forecasting.
SIH26153, National Technical Research Organisation (NTRO).

## Slide 2 — Reactive vs Proactive
- Traditional IDS: inspects isolated packets, alerts *after* breach.
- NetForecast: learns traffic dynamics over time, predicts an attack
  *before* compromise — past → present → future forecast.

## Slide 3 — State Representation
- Every 30 seconds of traffic is aggregated into one fixed-length state
  vector: flow-level features + packet-level features merge into
  kill-chain signatures, producing `S_t ∈ R^33` per window.

## Slide 4 — World Model Architecture
- Last 10 network snapshots → Encoder → LSTM Memory → Attention →
  3 prediction heads sharing one recurrent context: next-state (MSE),
  attack-stage (cross-entropy), infiltration probability (BCE).

## Slide 5 — K-Step Rollout & Kill-Chain Mapping
- The model simulates future network states t+1 … t+5, each prediction
  fed back as input — infiltration probability rises as the trajectory
  approaches the predicted MITRE ATT&CK kill-chain stage (Reconnaissance
  → Initial Access → Lateral Movement → C2 → Exfiltration).

## Slide 6 — Explainability
- **Attention weights** — which of the last observed time windows
  drove the forecast (e.g. "port scan detected here").
- **Feature saliency** — which specific traffic signals (scan ratio,
  SYN rate, beacon regularity, TTL variance) triggered the warning.

## Slide 7 — Results & Impact
- Benchmarked against a logistic-regression baseline on the **identical**
  context window, horizon and chronologically-held-out (future) test
  split on real CIC-IDS2017 traffic — see `reports/benchmark_cicids2017.md`.
- World model: F1 0.832 vs baseline 0.447, FPR 0.075 vs 0.828 — nearly
  double the F1 with an order of magnitude fewer false positives.

## Slide 8 — Key Features & Technology Stack
- Multi-task world model (dynamics + stage + infiltration heads),
  K-step forecasting, built-in explainability, offline demo,
  baseline-benchmarked.
- Stack: PyTorch, scikit-learn, SHAP, pandas/NumPy, Streamlit, Python.

## Slide 9 — Future Scope
- Graph-based per-host state with a GNN encoder for larger topologies.
- Streaming ingest (Kafka/NetFlow) for real-time deployment.
- Additional dataset adapters (CIC-IDS2018, CTU-13, CICIoT2023).
- Direct SIEM/SOAR integration for analyst alerting.
