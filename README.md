# NetForecast — AI World Model for Network Attack Forecasting

## 1. Project Information

- **Project Title:** NetForecast — AI World Model for Network Attack Forecasting
- **PS ID:** SIH26153
- **PS Title:** AI based Network Attack Forecasting from Network Traffic Data
- **Category:** Software
- **Organisation:** National Technical Research Organisation (NTRO)

## 2. Problem Statement

Static intrusion classifiers score each network flow in isolation as
benign or malicious. This discards the temporal and causal structure of
a real attack — the ordered sequence of port scanning, exploitation,
lateral movement, and command-and-control beaconing that make up an
actual intrusion. As a result, defenders are typically alerted only
after individual malicious flows are already captured, often too late
to prevent compromise.

## 3. Proposed Solution

NetForecast learns the evolving state of a network from traffic
telemetry and forecasts the **probability and progression of an attack
before compromise completes**, mapping predicted behaviour onto MITRE
ATT&CK stages, with built-in explainability (attention weights +
feature saliency/SHAP).

Unlike a static flow classifier, this is a **World Model**: it learns
the transition dynamics `P(S_t+1 | S_t)` of the network state itself,
then performs a K-step forward rollout to simulate where the current
trajectory is heading — flagging an attack while it is still unfolding,
not after.

## 4. Key Features

- Windowed flow + packet-level feature pipeline with engineered kill-chain
  signatures (scan ratio, beacon regularity, admin-port ratio, ...).
- LSTM **world model** that learns network-state transition dynamics,
  not just a per-flow label.
- K-step forward rollout — a forecasted infiltration-probability
  *trajectory*, plus predicted MITRE ATT&CK stage per step.
- Built-in explainability: attention weights + gradient×input saliency
  (and SHAP for the baseline).
- Offline Streamlit demo with real (CIC-IDS2017) and synthetic checkpoints.
- Benchmarked against a logistic-regression baseline on an identical,
  chronologically held-out test split — see [docs/architecture.md](docs/architecture.md).

## 5. Technology Stack

- **ML / modelling:** PyTorch (LSTM world model), scikit-learn (baseline),
  SHAP (explainability)
- **Data:** pandas, NumPy, CIC-IDS2017 (real) + custom synthetic generator
- **Demo UI:** Streamlit
- **Language:** Python

## 6. Architecture

```
Flow records (CSV/PCAP)
        │
        ▼
Feature extraction (src/features.py)
  - flow-level: 5-tuple, TCP flags, byte/pkt counts, IAT stats
  - packet-level: TTL variance, window size, retransmissions, fragmentation
  - windowed (30s) aggregation → fixed-length state vector S_t
        │
        ▼
LSTM World Model (src/world_model.py)
  - Linear encoder → 2-layer LSTM → additive attention over context
  - 3 heads: next-state (dynamics regression), attack-stage (classification),
    infiltration probability (binary, K-step horizon)
        │
        ▼
Infiltration Prediction Engine (src/predict.py)
  - K-step autoregressive rollout (model's own prediction fed back as input)
  - MITRE ATT&CK stage mapping (src/mitre_mapping.py)
  - Explainability: attention weights + gradient×input saliency
        │
        ▼
Streamlit demo (demo/app.py) — offline, CSV in → probability timeline,
flagged flows, attack-stage annotations, explanations out
```

See [docs/architecture.md](docs/architecture.md) for the full write-up
— state representation, benchmark methodology and results, dataset
adapter details, and honesty/scope notes.

## 7. Repository Structure

```text
netforecast/
├── README.md
├── SUBMISSION_GUIDE.md
├── LICENSE
├── submission/
│   ├── PRESENTATION.md        # SIH presentation
│   └── DEMO.md                # demo video link
├── docs/
│   └── architecture.md        # full architecture write-up
├── assets/
│   └── screenshots/           # demo screenshots
├── src/                       # feature pipeline, world model, training, evaluation
├── demo/app.py                 # Streamlit offline demo UI (switch real/synthetic in sidebar)
├── data/                       # raw + processed flow data (git-ignored, large)
├── models/ · models_real/       # trained checkpoints (synthetic / real CIC-IDS2017)
├── reports/                     # generated benchmark reports
└── requirements.txt
```

### What goes where?

| Item | Location |
|---|---|
| Source code | `src/`, `demo/` |
| Architecture / technical documentation | `docs/architecture.md` |
| Project screenshots | `assets/screenshots/` |
| Final PPT / presentation | `submission/` |
| Demo video link | `submission/DEMO.md` |
| Project overview | `README.md` |

## 8. Final Presentation

See [submission/PRESENTATION.md](submission/PRESENTATION.md) for the
SIH presentation.

## 9. Demo Video

[Watch on Google Drive](https://drive.google.com/drive/folders/14LbfeSOyGeUZZ3lBmPQWUZCFh_gERLxc?usp=sharing)

See [submission/DEMO.md](submission/DEMO.md) for the link.

## 10. Screenshots / Prototype Photos

### Ingested traffic overview
Flow count, time-window count, and window size for the loaded file.

![Traffic overview](assets/screenshots/01-overview.png)

### Live Replay Simulation
The risk gauge and MITRE kill-chain stepper updating window-by-window, as if the traffic were arriving live.

![Live Replay Simulation](assets/screenshots/03-live-replay-gauge.png)

### Explainability — top driving features
Top driving features (gradient×input saliency) behind the current risk score.

![Explainability — top driving features](assets/screenshots/07-saliency.png)

More screenshots (live replay result, K-step rollout, attention
weights, flagged flows, MITRE stage reference) are in
[assets/screenshots/](assets/screenshots/).

## 11. Installation

```bash
git clone https://github.com/ROHIT-25607/netforecast.git
cd netforecast
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 12. Run

### Run the demo

```bash
streamlit run demo/app.py
```

Upload a flow-record CSV (or leave blank to use the bundled sample).
The app shows the traffic overview, a rolling infiltration-probability
timeline, a K-step forward simulation with predicted MITRE ATT&CK
stage, attention/saliency explainability charts, and flagged flows for
analyst drill-down — fully offline, no external network or cloud-API
calls.

### Reproduce training + benchmark (synthetic data)

```bash
python src/simulate_traffic.py --out data/synthetic_flows.csv \
       --n-benign 20000 --n-campaigns 15 --duration-hours 48
python src/train.py --data data/synthetic_flows.csv --epochs 25
python src/baseline.py --data data/synthetic_flows.csv
python src/evaluate.py   # -> writes reports/benchmark.md
```

### Train on the real CIC-IDS2017 dataset

```bash
# put the 8 daily CIC-IDS2017 CSVs in data/raw/, then:
python src/real_data_adapter.py --raw-dir data/raw \
       --out data/cicids2017_processed.csv --window-flow-count 200
python src/train.py --data data/cicids2017_processed.csv \
       --window-seconds 200 \
       --day-boundaries data/cicids2017_processed.day_boundaries.json \
       --out-dir models_real --epochs 25
python src/baseline.py --data data/cicids2017_processed.csv \
       --model-dir models_real --window-seconds 200
```

See [docs/architecture.md](docs/architecture.md) for benchmark results
and adapter details.

## 13. Future Scope

- Per-host graph state with a GNN encoder for larger enterprise topologies.
- Streaming ingest (Kafka/NetFlow collector feed) instead of CSV batch files.
- Adapters for additional real-world datasets (CIC-IDS2018, CTU-13, CICIoT2023).
- Direct SIEM/SOAR integration for analyst alerting.
