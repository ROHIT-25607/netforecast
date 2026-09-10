# Project Presentation

**PPT:** [Open Final Presentation](./NetForecast_SIH2026_Presentation.pptx)

7 slides: Title → Problem & Approach → Architecture → Key Features &
Tech Stack → Results & Impact → Explainability & MITRE ATT&CK Mapping →
Future Scope.

## External presentation link (only if needed)

Not required — the PPTX above is small enough to keep in the repository.

---

## Slide-by-slide notes (for narration)

## Slide 1 — Title
NetForecast — AI World Model for Network Attack Forecasting.
SIH26153, National Technical Research Organisation (NTRO).

## Slide 2 — Problem & Approach
- Static flow classifiers score each packet/flow in isolation — they miss the
  *sequence* of an infiltration (scan → exploit → pivot → beacon → exfil).
- **NetForecast** learns a **World Model**: the transition dynamics
  `P(S_t+1 | S_t)` of the network's own state, then simulates forward to
  catch attacker progression *before* compromise completes.
- Fully offline, open-source prototype: feature pipeline → LSTM world model
  → K-step forecaster → MITRE ATT&CK mapping → Streamlit UI.

## Slide 3 — System Architecture
- Flow records → feature extraction (windowed 33-feature state `S_t`,
  flow-level + packet-level + engineered kill-chain signatures) → LSTM
  world model (encoder + attention) → prediction engine (K-step rollout)
  → analyst dashboard.
- Every prediction carries a MITRE ATT&CK stage and an explanation.

## Slide 4 — Key Features & Technology Stack
- Multi-task world model (dynamics + stage + infiltration heads),
  K-step forecasting, built-in explainability, offline demo,
  baseline-benchmarked.
- Stack: PyTorch, scikit-learn, SHAP, pandas/NumPy, Streamlit, Python.

## Slide 5 — Results & Impact
- Benchmarked against a logistic-regression baseline on the **identical**
  context window, horizon and chronologically-held-out (future) test
  split on real CIC-IDS2017 traffic — see `reports/benchmark_cicids2017.md`.
- World model: F1 0.832 vs baseline 0.447, FPR 0.075 vs 0.828 — nearly
  double the F1 with an order of magnitude fewer false positives.

## Slide 6 — Explainability & MITRE ATT&CK Mapping
- Every prediction carries **attention weights** (which recent time
  windows mattered) + **gradient×input saliency** (which traffic
  features — SYN rate, scan ratio, beacon regularity, TTL variance —
  drove the score). SHAP cross-checks the baseline.
- Predicted stage is mapped to its MITRE ATT&CK Enterprise tactic ID
  (TA0043 Reconnaissance, TA0001 Initial Access, TA0008 Lateral Movement,
  TA0011 C2, TA0010 Exfiltration) for direct SOC/analyst consumption.

## Slide 7 — Future Scope
- Graph-based per-host state with a GNN encoder for larger topologies.
- Streaming ingest (Kafka/NetFlow) for real-time deployment.
- Additional dataset adapters (CIC-IDS2018, CTU-13, CICIoT2023).
- Direct SIEM/SOAR integration for analyst alerting.
