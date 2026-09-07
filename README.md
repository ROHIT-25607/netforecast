# NetForecast — AI World Model for Network Attack Forecasting

**SIH Problem Statement 26153** — National Technical Research Organisation (NTRO)
*AI based Network Attack Forecasting from Network Traffic Data*

NetForecast learns the evolving state of a network from traffic telemetry and
forecasts the **probability and progression of an attack before compromise
completes**, mapping predicted behaviour onto MITRE ATT&CK stages, with
built-in explainability (attention weights + feature saliency/SHAP).

Unlike a static flow classifier (benign/malicious per flow), this is a
**World Model**: it learns the transition dynamics `P(S_t+1 | S_t)` of the
network state itself, then performs K-step forward rollout to simulate
where the current trajectory is heading.

---

## 1. Architecture at a glance

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

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full write-up.

## 2. Repository layout

```
netforecast/
├── src/
│   ├── simulate_traffic.py   # synthetic CIC-IDS2018-schema flow generator (kill-chain campaigns)
│   ├── features.py           # flow → windowed state-vector pipeline
│   ├── dataset.py            # sequence windowing for supervised dynamics learning
│   ├── world_model.py        # LSTM + attention world model (PyTorch)
│   ├── train.py               # multi-task training (dynamics + stage + infiltration)
│   ├── baseline.py            # logistic-regression benchmark baseline
│   ├── evaluate.py            # F1/precision/recall/FPR benchmark, world model vs baseline
│   ├── predict.py             # K-step rollout + MITRE mapping + explainability
│   ├── explain.py             # SHAP explainability for the baseline
│   └── mitre_mapping.py       # attack-stage ↔ MITRE ATT&CK tactic mapping
├── demo/app.py                # Streamlit offline demo UI
├── data/                      # generated/uploaded flow CSVs
├── models/                    # trained checkpoints, config, norm stats
├── reports/benchmark.md       # generated benchmark report
└── requirements.txt
```

## 3. Setup

```bash
cd netforecast
python -m venv .venv && source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## 4. Reproduce end-to-end (training + benchmark)

```bash
# 1. Generate the labelled synthetic traffic dataset (kill-chain campaigns
#    interleaved with benign background traffic; ~24-48h of simulated flows)
python src/simulate_traffic.py --out data/synthetic_flows.csv \
       --n-benign 20000 --n-campaigns 15 --duration-hours 48

# 2. Train the LSTM world model (multi-task: dynamics + stage + infiltration)
python src/train.py --data data/synthetic_flows.csv --epochs 25

# 3. Train the logistic-regression baseline on the identical features/split
python src/baseline.py --data data/synthetic_flows.csv

# 4. Benchmark: F1 / precision / recall / FPR, world model vs baseline
python src/evaluate.py
# -> writes reports/benchmark.md

# 5. (optional) SHAP explanation of the baseline, for contrast
python src/explain.py
```

All scripts are deterministic (fixed seed 42) and the exact chronological
train/val/test split indices are cached in `models/split_indices.npz` so
`baseline.py` and `evaluate.py` compare against the *same* held-out,
future-in-time windows the world model never trained on.

## 5. Run the offline demo

```bash
streamlit run demo/app.py
```

Upload a flow-record CSV (or leave blank to use the bundled sample) and the
app will show:

1. Traffic overview (flow count, window count).
2. A rolling 1-step-ahead infiltration-probability timeline across the
   whole file.
3. A K-step forward simulation from the *current* (most recent) state:
   per-step infiltration probability + predicted MITRE ATT&CK stage.
4. Attention weights over the recent context window (explainability).
5. Top driving traffic features for the current risk score (gradient×input
   saliency).
6. The raw flows in the most recent time window, for analyst drill-down.

The app never calls out to the network or any cloud API — inference is a
local forward pass through the checkpoint in `models/`.

## 6. Using a real dataset (CIC-IDS2018 / CTU-13 / CICIoT2023)

`src/features.py` expects the column schema documented in
`simulate_traffic.COLUMNS`. To plug in a real CICFlowMeter export:

1. Rename that dataset's columns to match (e.g. CIC-IDS2018's
   `Flow Duration` → `duration`, `Tot Fwd Pkts` → `tot_fwd_pkts`,
   `SYN Flag Cnt` → `syn_flag_cnt`, `TotLen Fwd Pkts` → `totlen_fwd_bytes`,
   etc. — a 1:1 mapping for almost every field).
2. Derive `label_stage` from the dataset's attack-timeline annotations
   (CIC-IDS2018 publishes per-day attack windows/labels; map each label to
   one of `Benign / Reconnaissance / Initial_Access / Lateral_Movement /
   Command_And_Control / Exfiltration` per `mitre_mapping.STAGES`).
3. For packet-level fields not present in a flow-only export (TTL
   variance, retransmission count, fragmentation), parse the corresponding
   PCAP with **Scapy** or **PyShark** and join on the flow 5-tuple + time
   window; a short adapter script following the `_flow` builders in
   `simulate_traffic.py` is the fastest way to do this.
4. Run `train.py --data <your.csv>` — no other code changes needed.

## 7. Why a World Model instead of a classifier?

A per-flow classifier scores each flow independently and cannot express
"the last 10 minutes of scanning + this new SMB connection means lateral
movement is imminent." NetForecast instead:

- Represents the **whole monitored network's state** at each 30s window
  as a single vector (flow volumes, flag mixes, scan/beacon/exfil
  signatures — see `features.STATE_FEATURE_NAMES`).
- Learns **P(S_t+1 | S_t)** — i.e. it is trained to predict the *next*
  state vector, not just today's label — via an MSE dynamics loss plus
  auxiliary stage/infiltration heads sharing the same recurrent context.
- **Simulates forward** K steps by re-feeding its own predictions,
  producing a probability *trajectory*, not a single snapshot score.
- `reports/benchmark.md` quantifies the resulting gain over a
  logistic-regression baseline given the identical context window and
  forecast horizon.

## 8. Explainability

- **Attention weights** (`world_model.AdditiveAttention`) show which of
  the last L observed time windows the model weighted most heavily.
- **Gradient × input saliency** (`predict.InfiltrationPredictor._saliency`)
  ranks which of the ~30 traffic features (SYN rate, scan ratio, beacon
  regularity, TTL variance, admin-port ratio, …) drove a given
  infiltration score.
- **SHAP** (`explain.py`) is used on the logistic-regression baseline as
  an independent, model-agnostic cross-check of feature importance.

No prediction is surfaced without at least one of these attached — this
was a hard requirement in the problem statement.

## 9. Limitations / honesty notes

- Training data is a **synthetic generator** (see §6) built to match the
  CIC-IDS2018 flow schema and known kill-chain signatures, because the
  real multi-GB datasets could not be fetched inside this environment.
  The pipeline, model and demo are dataset-agnostic and load a real
  export unchanged once columns are mapped (§6).
- The "network state" here is a single aggregated vector for the whole
  monitored segment per time window. A natural extension (noted in
  `ARCHITECTURE.md`) is a per-host graph state with a GNN encoder for
  larger enterprise topologies.
- K-step rollout accumulates model error autoregressively, as in any
  latent-dynamics/world-model rollout; `evaluate.py` reports metrics at
  the trained horizon K to keep this honest rather than cherry-picking
  short horizons.
