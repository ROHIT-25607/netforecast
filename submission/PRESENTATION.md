# Technical Presentation — NetForecast (5 slides)
*Paste each `##` section into one slide.*

## Slide 1 — Problem & Approach
**AI-based Network Attack Forecasting from Network Traffic Data (SIH 26153, NTRO)**
- Static flow classifiers score each packet/flow in isolation — they miss the
  *sequence* of an infiltration (scan → exploit → pivot → beacon → exfil).
- **NetForecast** learns a **World Model**: the transition dynamics
  `P(S_t+1 | S_t)` of the network's own state, then simulates forward to
  catch attacker progression *before* compromise completes.
- Fully offline, open-source prototype: feature pipeline → LSTM world model
  → K-step forecaster → MITRE ATT&CK mapping → Streamlit UI.

## Slide 2 — State Representation & Data
- Network state `S_t`: 33-feature vector per 30s window, combining
  **flow-level** (5-tuple, TCP flags, byte/pkt counts, IAT) and
  **packet-level** (TTL variance, window size, retransmissions,
  fragmentation) statistics, plus engineered kill-chain signatures
  (scan ratio, half-open ratio, beacon regularity, admin-port ratio).
- Trained on a labelled flow-record dataset built to the CIC-IDS2018
  schema, with injected full kill-chain campaigns (Reconnaissance →
  Initial Access → Lateral Movement → C2 → Exfiltration) interleaved with
  benign background traffic — pipeline is dataset-agnostic and accepts a
  real CIC-IDS2018 / CTU-13 export via a column-mapping adapter.

## Slide 3 — World Model Architecture
- Linear encoder → 2-layer LSTM → **additive attention** over the last
  L=10 windows → 3 heads sharing one recurrent context:
  1. **Next-state** (regression, MSE) — the actual dynamics-learning signal
  2. **Attack-stage** classification (6-way, cross-entropy)
  3. **Infiltration probability** (binary, K-step horizon, BCE)
- **K-step rollout**: model's own predicted next state is fed back as
  input for K steps → a full infiltration-probability *trajectory*, not a
  single score.

## Slide 4 — Explainability & MITRE ATT&CK Mapping
- Every prediction carries **attention weights** (which recent time
  windows mattered) + **gradient×input saliency** (which traffic
  features — SYN rate, scan ratio, beacon regularity, TTL variance —
  drove the score). SHAP cross-checks the baseline.
- Predicted stage is mapped to its MITRE ATT&CK Enterprise tactic ID
  (TA0043 Reconnaissance, TA0001 Initial Access, TA0008 Lateral Movement,
  TA0011 C2, TA0010 Exfiltration) for direct SOC/analyst consumption.

## Slide 5 — Results & Impact
- Benchmarked against a logistic-regression baseline on the **identical**
  context window, horizon and chronologically-held-out (future) test
  split — see `reports/benchmark.md` for exact figures.
- World model shows measurable F1/recall improvement, evidencing that
  learned temporal dynamics — not just more features — drive the gain.
- Applicable to enterprise SOC and Critical Information Infrastructure
  monitoring: offline deployment, no cloud dependency, interpretable
  output an analyst can act on before the kill chain completes.
