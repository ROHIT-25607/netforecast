# NetForecast — AI World Model for Network Attack Forecasting

## 1. Project Information

- **Project Title:** NetForecast — AI World Model for Network Attack Forecasting
- **PS ID:** SIH26153
- **PS Title:** AI based Network Attack Forecasting from Network Traffic Data
- **Category:** Software
- **Organisation:** National Technical Research Organisation (NTRO)

### Team

| Name | Role | Responsibilities |
|---|---|---|
| Praneel Maumdar | Team lead / ML | World model architecture, training, evaluation |
| Rohit Kumar Singh | Backend | FastAPI service, WebSocket streaming, live ingest |
| Amit Patel | Data engineering | Feature pipeline, CIC-IDS2017 adapter, synthetic generator |
| Himanshu Yadav | Frontend | SOC dashboard, visualization |
| Keshav Mittal | Security research | MITRE ATT&CK mapping, threat-model validation |
| Nikita Gupta | Documentation / QA | Benchmarks, tests, submission materials |

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
- **FastAPI service + SOC-console dashboard**, with a WebSocket replay and a
  `POST /api/ingest` endpoint that scores live flow batches from a collector.
- Runs **fully air-gapped**: model, dashboard assets and sample capture are all
  local; the running service makes no outbound network call.
- Benchmarked against logistic-regression **and random-forest** baselines on an
  identical held-out split, with ROC/PR curves, a **forecast-vs-horizon** table,
  **lead-time** statistics and a rollout ablation — see
  [reports/benchmark_cicids2017.md](reports/benchmark_cicids2017.md).

## 5. Technology Stack

- **ML / modelling:** PyTorch (LSTM world model), scikit-learn (baseline),
  SHAP (explainability)
- **Data:** pandas, NumPy, CIC-IDS2017 (real) + custom synthetic generator
- **Service:** FastAPI + Uvicorn (REST + WebSocket), vanilla-JS dashboard, Plotly
- **Language:** Python 3.10+

## 6. Architecture

```
Flow records (CSV)          Live collector (NetFlow / Zeek / Kafka)
        │                            │
        │                            ▼
        │                   POST /api/ingest  ──┐
        ▼                                       │
Feature extraction (netforecast/features.py)  ◄─┘
  - flow-level: 5-tuple, TCP flags, byte/pkt counts, IAT stats
  - packet-level: TTL variance, window size, retransmissions, fragmentation
  - windowed aggregation → fixed-length 33-dim state vector S_t
    (fully vectorized: ~280x faster than the original per-window loop)
        │
        ▼
LSTM World Model (netforecast/world_model.py)
  - Linear encoder → 2-layer LSTM → additive attention over context
  - 3 heads: next-state (dynamics regression), attack-stage (classification),
    infiltration probability (binary, K-step horizon)
        │
        ▼
Infiltration Prediction Engine (netforecast/predict.py)
  - K-step autoregressive rollout (model's own prediction fed back as input)
  - batched scoring: a whole capture's timeline in one forward pass
  - MITRE ATT&CK stage mapping (netforecast/mitre_mapping.py)
  - Explainability: attention weights + gradient×input saliency
        │
        ▼
FastAPI service (server/)
  - REST:      /api/session, /timeline, /forecast, /explain, /flows, /state
  - WebSocket: /ws/replay/{id}  (streamed window-by-window replay)
               /ws/live/{src}   (live-ingest push)
  - OpenAPI docs at /docs
        │
        ▼
SOC dashboard (server/static) — risk gauge, MITRE kill-chain stepper,
probability timeline, K-step forecast, explainability, flow drill-down
```

See [docs/architecture.md](docs/architecture.md) for the full write-up
— state representation, benchmark methodology and results, dataset
adapter details, and honesty/scope notes.

**Headline results** (real CIC-IDS2017 traffic, 1,839 held-out windows):
F1 **0.843**, ROC-AUC **0.940**, PR-AUC **0.907**; the 5-step autoregressive
rollout retains **92%** of single-shot F1 while running four of its five steps
on self-generated state. Full tables, ROC/PR curves, per-horizon forecast
quality, lead time and latency are in
[reports/benchmark_cicids2017.md](reports/benchmark_cicids2017.md).

## 7. Repository Structure

```text
netforecast/
├── README.md · SUBMISSION_GUIDE.md · LICENSE
├── pyproject.toml · requirements.txt · requirements-dev.txt
├── Dockerfile · docker-compose.yml · Makefile · run.ps1
├── netforecast/                # the package: features, world model, training, evaluation
│   ├── features.py             #   flow -> 33-dim windowed state (vectorized)
│   ├── world_model.py          #   LSTM + additive attention, 3 heads
│   ├── predict.py              #   rollout + batched scoring + explainability
│   ├── train.py · baseline.py · evaluate.py
│   ├── dataset.py · mitre_mapping.py
│   └── real_data_adapter.py · simulate_traffic.py
├── server/                     # FastAPI service
│   ├── main.py · config.py · session.py · schemas.py · ingest.py
│   ├── routers/{api,stream}.py
│   └── static/                 #   dashboard (+ vendored Plotly, so it runs offline)
├── tools/
│   ├── feeder.py               #   replays a CSV into /api/ingest (stand-in collector)
│   └── fetch_vendor.py         #   one-time fetch of offline dashboard assets
├── tests/                      # pytest suite incl. the feature golden-value lock
├── docs/architecture.md        # full technical write-up
├── data/                       # sample_flows.csv committed; large captures git-ignored
├── models/ · models_real/      # trained checkpoints (committed — a clone can run as-is)
├── reports/                    # generated benchmarks + ROC/PR curves
└── assets/screenshots/
```

### What goes where?

| Item | Location |
|---|---|
| ML package | `netforecast/` |
| API + dashboard | `server/` |
| Architecture / technical documentation | `docs/architecture.md` |
| Benchmarks and curves | `reports/` |
| Tests | `tests/` |
| Project screenshots | `assets/screenshots/` |
| Final PPT / presentation | `submission/` |
| Demo video link | `submission/DEMO.md` |

## 8. Final Presentation

See [submission/PRESENTATION.md](submission/PRESENTATION.md) for the
SIH presentation.

## 9. Demo Video

[Watch on Google Drive](https://drive.google.com/drive/folders/14LbfeSOyGeUZZ3lBmPQWUZCFh_gERLxc?usp=sharing)

See [submission/DEMO.md](submission/DEMO.md) for the link.

## 10. Screenshots / Prototype Photos

### The dashboard

![NetForecast dashboard](assets/screenshots/00-dashboard.png)

### Current risk and MITRE kill chain

The gauge shows peak infiltration risk across the forecast horizon; the stepper
shows how far along the kill chain the predicted progression reaches.

![Risk gauge and kill chain](assets/screenshots/03-live-replay-gauge.png)

### Replay

The capture streamed window-by-window over a WebSocket. The rest of the page
stays interactive throughout.

![Live replay](assets/screenshots/04-live-replay-result.png)

### Live ingest

Flows arriving over `POST /api/ingest` from an external collector, scored and
pushed to the dashboard in real time. The flat stretches are benign traffic; the
plateaus are active campaign windows.

![Live ingest](assets/screenshots/10-live-ingest.png)

### Explainability — top driving features

Gradient×input saliency over the 33-dim state vector behind the current score.

![Top driving features](assets/screenshots/07-saliency.png)

The remaining screens (traffic overview, probability timeline, K-step rollout
table, attention weights, flow drill-down, MITRE reference) are in
[assets/screenshots/](assets/screenshots/).

## 11. Installation

```bash
git clone https://github.com/ROHIT-25607/netforecast.git
cd netforecast
python -m venv .venv && source .venv/bin/activate   # .venv\Scriptsctivate on Windows
pip install -r requirements.txt
```

The trained checkpoints and a small sample capture are committed, so a fresh
clone runs immediately — no dataset download and no training required.

The dashboard's charting library is vendored under `server/static/vendor/` so the
tool works on an air-gapped network. If it is missing, fetch it once with
`python tools/fetch_vendor.py`.

**Docker** (no local Python needed):

```bash
docker compose up --build     # dashboard on http://localhost:8000
```

The image bundles the checkpoints, the dashboard assets and the sample capture,
and the running container makes no outbound network calls — the same air-gapped
posture the on-prem deployment assumes. The large CIC-IDS2017 capture is *not*
baked in (396 MB); mount it with the `./data` volume in `docker-compose.yml` to
score it inside the container.

## 12. Run

### Start the service

```bash
uvicorn server.main:app --reload
# or:  make serve      /      .
un.ps1
```

| What | Where |
|---|---|
| SOC dashboard | <http://127.0.0.1:8000> |
| Interactive API docs (OpenAPI) | <http://127.0.0.1:8000/docs> |
| Health check | <http://127.0.0.1:8000/api/health> |

Pick a capture, press **Load capture**, and the dashboard shows the traffic
overview, the infiltration-probability timeline, a K-step forward simulation with
predicted MITRE ATT&CK stage, attention/saliency explainability, and per-window
flow drill-down. **Play replay** streams the capture window-by-window over a
WebSocket while the rest of the page stays interactive.

### Demonstrate the live path

With the server running, in a second terminal:

```bash
python tools/feeder.py --csv data/sample_flows.csv --rate 300
```

The feeder POSTs flow batches to `/api/ingest`; the server windows and scores
them and pushes results to the dashboard's **Live ingest** panel. Point a real
NetFlow/IPFIX collector, a Zeek `conn.log` tail or a Kafka consumer at the same
endpoint and nothing downstream changes.

### Reproduce the benchmark

```bash
make eval          # synthetic  -> reports/benchmark.md
make eval-real     # CIC-IDS2017 -> reports/benchmark_cicids2017.md
```

### Retrain from scratch (optional)

```bash
python -m netforecast.simulate_traffic --out data/synthetic_flows.csv        --n-benign 20000 --n-campaigns 15 --duration-hours 48
python -m netforecast.train --data data/synthetic_flows.csv --epochs 25
python -m netforecast.baseline --data data/synthetic_flows.csv
python -m netforecast.evaluate
```

On the real CIC-IDS2017 dataset (put the 8 daily CSVs in `data/raw/` first):

```bash
python -m netforecast.real_data_adapter --raw-dir data/raw        --out data/cicids2017_processed.csv --window-flow-count 200
python -m netforecast.train --data data/cicids2017_processed.csv        --window-seconds 200        --day-boundaries data/cicids2017_processed.day_boundaries.json        --out-dir models_real --epochs 25
python -m netforecast.baseline --data data/cicids2017_processed.csv        --model-dir models_real --window-seconds 200
python -m netforecast.evaluate --model-dir models_real        --data data/cicids2017_processed.csv        --report reports/benchmark_cicids2017.md --window-unit flows
```

### Tests

```bash
make test     # pytest
make lint     # ruff
```

## 13. Future Scope

- Per-host graph state with a GNN encoder for larger enterprise topologies.
- Kafka / NetFlow collector adapters in front of the existing `/api/ingest`
  endpoint (the ingestion seam and its rolling-window scorer already exist).
- Adapters for additional real-world datasets (CIC-IDS2018, CTU-13, CICIoT2023).
- Direct SIEM/SOAR integration for analyst alerting.
