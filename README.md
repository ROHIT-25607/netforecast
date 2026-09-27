# NetForecast — AI World Model for Network Attack Forecasting

## 1. Project Information

- **Project Title:** NetForecast — AI World Model for Network Attack Forecasting
- **PS ID:** SIH26153
- **PS Title:** AI based Network Attack Forecasting from Network Traffic Data
- **Category:** Software
- **Organisation:** National Technical Research Organisation (NTRO)

### Team

<!-- TODO(before submission): replace these placeholder rows with the real team.
     The SIH checklist requires team members and roles to be named. -->

| Name | Role | Responsibilities |
|---|---|---|
| _TBD_ | Team lead / ML | World model architecture, training, evaluation |
| _TBD_ | Data engineering | Feature pipeline, CIC-IDS2017 adapter, synthetic generator |
| _TBD_ | Backend | FastAPI service, WebSocket streaming, live ingest |
| _TBD_ | Frontend | SOC dashboard, visualization |
| _TBD_ | Security research | MITRE ATT&CK mapping, threat-model validation |
| _TBD_ | Documentation / QA | Benchmarks, tests, submission materials |

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

## 7. Results

Full reports, including ROC/PR curves, live in [`reports/`](reports/). Every number
below comes from the committed checkpoints on a held-out split; reproduce with
`make eval` / `make eval-real`.

### Detection — CIC-IDS2017 (1,839 held-out windows, 32.3% positive)

| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.311 | 0.783 | 0.445 | 0.829 | 0.452 | 0.413 |
| Random Forest | 0.950 | 0.665 | 0.782 | **0.017** | **0.948** | 0.905 |
| **LSTM World Model** | 0.894 | 0.798 | **0.843** | 0.045 | 0.940 | **0.907** |

The world model gives the best F1 (**+0.061** over the strongest baseline) by being
better balanced: the random forest is highly precise but misses 33% of attack
windows, which in a SOC is the expensive kind of error. The two are now level on
ranking quality (ROC-AUC 0.940 vs 0.948, PR-AUC 0.907 vs 0.905).

### Forecast quality vs horizon

Step *s* is the **autoregressive rollout**: the model has fed its own predicted
state back in *s−1* times and has seen no new traffic.

| Horizon | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| F1 | 0.849 | 0.841 | 0.815 | 0.794 | 0.775 |
| ROC-AUC | 0.953 | 0.946 | 0.936 | 0.929 | 0.923 |

Predicting the same label from a single forward pass over *observed* traffic scores
F1 0.843. Running 4 of its 5 steps on self-generated state, the rollout retains
**92%** of that. A model that had merely memorised a current-window mapping would
collapse once fed its own output — this is the evidence the transition function is
real.

### Lead time

Of 40 attack-episode onsets assessable in the held-out period, the model was already
above threshold **before the first malicious flow was recorded** in 18 (**45%**),
with a median warning of 1 window (200 flows) and a maximum of 3 windows (600 flows).

### Latency (CPU, single process)

| Operation | Median |
|---|---|
| Featurize 5,000 flows | 18.8 ms |
| Single forward pass | 3.01 ms |
| 5-step rollout | 9.12 ms |
| Rollout + explainability | 9.08 ms |

No GPU and no network call anywhere in the inference path.

### Training and model selection

The CIC-IDS2017 checkpoint was retrained for the final round. What actually moved the
numbers was ordinary training hygiene rather than anything exotic:

| Change | Effect |
|---|---|
| `ReduceLROnPlateau` + gradient clipping + early stopping | removed the val-loss spikes (0.82 → 2.10 mid-run) visible in the old `history.json` |
| Restore **best-validation** weights before saving | the previous run wrote `config.json`/`history.json` from the last epoch |
| Corrected `day_aware_split` window indices | see the caveat below |

Two changes that sounded promising were **measured and rejected**:

- **Re-weighting the loss.** At the old weights the 33-dim dynamics MSE was ~70% of
  total loss and the infiltration head — the one every metric scores — only 8–12%.
  Up-weighting the BCE term made held-out PR-AUC *worse* (0.839 vs 0.907).
- **Multi-step (autoregressive) dynamics loss.** Intended to attack rollout drift
  directly; it came out level-to-slightly-worse than one-step training.

Both remain available behind `--w-dyn` / `--w-inf` / `--rollout-steps`, defaulted to
the configuration that actually won. Every candidate was selected on **validation**
and only then scored on test, and a candidate was promoted only if it beat the
incumbent on *every* held-out metric — which is why the synthetic checkpoint was
left alone. `tools/eval_seeds.py` and `tools/ab_compare.py` reproduce the comparison.

Against the previous checkpoint, on the identical held-out split:

| Metric | before | after |
|---|---|---|
| F1 | 0.832 | **0.843** |
| ROC-AUC | 0.898 | **0.940** |
| PR-AUC | 0.881 | **0.907** |
| 5-step rollout F1 | 0.670 | **0.775** |
| Rollout retention | 81% | **92%** |

### Honest caveats

These are stated up front because a reviewer will find them anyway, and they bound
what the numbers mean:

- **On synthetic data the world model does *not* beat a random forest** (F1 0.938 vs
  0.941). The generator is close to trivially separable, so the baselines saturate.
  The CIC-IDS2017 comparison is the meaningful one. Retraining the synthetic model
  with the improved recipe made it *worse* on every held-out metric, so the original
  checkpoint is what ships — see [Training and model selection](#training-and-model-selection).
- **The real-data checkpoint is the validation-selected seed of three.** Across
  seeds, held-out PR-AUC ranged 0.879–0.907. The shipped model is the best on
  *validation*; its test numbers are reported above and are reproducible, but a
  retrain from a different seed would not necessarily land in the same place.
- **~9 of the 33 state features are constant zeros on CIC-IDS2017.** TTL variance,
  retransmission and fragmentation are not recoverable from that export, and its
  adapter uses placeholder IPs. Saliency on the real model therefore ranks a narrower
  feature set than the synthetic model does.
- **`window_seconds: 200` for the real model counts flows, not seconds.** CIC-IDS2017
  has no usable wall clock, so the adapter uses row order. The API reports
  `window_unit` explicitly so nothing in the UI claims otherwise.
- **The Exfiltration stage has no training examples on CIC-IDS2017**, and DoS/DDoS
  rows are dropped by the adapter. The stage head can still emit Exfiltration; on the
  real checkpoint that output is not evidence-backed.
- **Best-F1 thresholds in the reports are selected on the test split**, so treat them
  as an upper bound; a deployment would tune on validation.
- The real-data split is `day_aware_split` (chronological *within* each capture day,
  then unioned), not one strictly-future cut — otherwise whole attack families would
  appear only in test. See [docs/architecture.md](docs/architecture.md) §6.

## 8. Repository Structure

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

## 9. Final Presentation

See [submission/PRESENTATION.md](submission/PRESENTATION.md) for the
SIH presentation.

## 10. Demo Video

[Watch on Google Drive](https://drive.google.com/drive/folders/14LbfeSOyGeUZZ3lBmPQWUZCFh_gERLxc?usp=sharing)

See [submission/DEMO.md](submission/DEMO.md) for the link.

## 11. Screenshots / Prototype Photos

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

## 12. Installation

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

## 13. Run

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

## 14. Future Scope

- Per-host graph state with a GNN encoder for larger enterprise topologies.
- Kafka / NetFlow collector adapters in front of the existing `/api/ingest`
  endpoint (the ingestion seam and its rolling-window scorer already exist).
- Adapters for additional real-world datasets (CIC-IDS2018, CTU-13, CICIoT2023).
- Direct SIEM/SOAR integration for analyst alerting.
