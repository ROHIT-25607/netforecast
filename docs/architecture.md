# Architecture Document — NetForecast
### AI World Model for Network Attack Forecasting (SIH 26153, NTRO)

## 1. Problem framing

Static intrusion classifiers map one flow to one label and discard the
temporal/causal structure of an infiltration — the order in which ports
are probed, the regularity of C2 beacon timing, the point at which
internal admin-protocol traffic starts appearing after a successful
exploit. NetForecast instead treats the monitored network as a **dynamical
system** with a state `S_t` that evolves over time, and learns the
transition function `P(S_t+1 | S_t)` — a **World Model** in the sense used
in model-based RL: a learned simulator of "what happens next," used here
for forward-looking cyber defence rather than control.

## 2. State representation

Every `WINDOW_SECONDS` (default 30s) of captured flow records is
aggregated into one fixed-length vector `S_t ∈ R^33` (`features.py`,
`STATE_FEATURE_NAMES`) combining:

**Flow-level** — flow count, unique src/dst counts, mean/std duration,
byte and packet counts (fwd/bwd), TCP flag rates (SYN/ACK/FIN/RST/PSH/URG),
forward IAT mean/std/max, bidirectional byte ratio.

**Packet-level (derived)** — mean TTL and TTL variance across the window,
initial TCP window size (fwd/bwd), retransmission rate, IP-fragmentation
rate.

**Engineered kill-chain signatures** — `scan_ratio` (unique destination
ports ÷ flow count — reconnaissance), `half_open_ratio` (SYN with no
ACK/FIN — half-open scan), `beacon_regularity` (inverse coefficient of
variation of inter-arrival time to the top destination — C2 beaconing),
`internal_internal_ratio` and `admin_port_ratio` (445/3389/5985/135 —
lateral movement), `external_dst_ratio`.

This vector is the "state" the world model predicts one step ahead. A
graph-based state (per-host nodes, flow edges, GNN encoder) is the natural
extension for larger enterprise topologies with many simultaneously active
hosts — noted as future work in §7.

## 3. World model

`world_model.WorldModel` (PyTorch):

```
S_{t-L+1..t} (L=10) → Linear encoder → 2-layer LSTM → h_1..h_L
                                                          │
                                        additive attention over h_1..h_L
                                                          │
                                                   context vector c
                        ┌─────────────────────────┬──────┴──────────────┐
                        │                         │                     │
                 next_state head           stage head          infiltration head
                (Linear→33, MSE)      (Linear→6 classes,   (Linear→1, BCE, K-step
                 dynamics loss          cross-entropy)       horizon label)
```

Trained end-to-end with a combined loss
`L = MSE(next_state) + CE(stage) + BCE(infiltration)`, so the recurrent
encoder is forced to capture dynamics rich enough to *reconstruct the next
raw state*, not merely to discriminate a label — this is what
distinguishes it from a plain sequence classifier and is what the
benchmark in §6 is designed to isolate.

The attention layer over the L context steps doubles as the primary
explainability signal (§5).

## 4. K-step forward simulation (rollout)

`predict.InfiltrationPredictor.rollout`: starting from the last observed
context window, the model produces `next_state`; that predicted state is
appended to the context (oldest step dropped) and fed back in for K
successive steps. At every step the stage and infiltration heads are read
out, producing:

- a **time-series infiltration probability** for t+1 .. t+K,
- a **predicted MITRE ATT&CK stage** per step (§5),
- the **overall infiltration risk** = max probability across the horizon.

This is what lets the system flag "this trajectory is heading toward
lateral movement in ~3 windows" *before* the corresponding flows are
observed, rather than only scoring flows already captured.

## 5. MITRE ATT&CK mapping and explainability

`mitre_mapping.py` maps each of the 6 stage labels
(Benign / Reconnaissance / Initial_Access / Lateral_Movement /
Command_And_Control / Exfiltration) to its MITRE ATT&CK Enterprise tactic
ID (TA0043, TA0001, TA0008, TA0011, TA0010) with a one-line rationale,
surfaced directly in the predicted-stage output and the demo UI.

Two independent, complementary explanations are attached to every
prediction — a hard requirement of the problem statement:

1. **Attention weights** over the L most recent observed windows — "which
   recent time steps mattered."
2. **Gradient × input saliency** over the ~33 state features of those
   windows — "which specific flags/ports/timing statistics mattered,"
   ranked and shown as a bar chart.

For the logistic-regression baseline, `explain.py` additionally computes
**SHAP** values (`shap.LinearExplainer`) as a model-agnostic cross-check.

## 6. Benchmark methodology & results

`baseline.py` trains **two** baselines on the *identical* flattened L-step
context window and K-step infiltration label used by the world model, so the
comparison isolates the value of the learned recurrent dynamics rather than
differences in features or horizon:

- a class-balanced **logistic regression** (the linear reference), and
- a **random forest** (a strong non-linear reference on the same 330-dim input).

The random forest matters. On CIC-IDS2017 the logistic regression is close to
degenerate — it fires on 83% of benign windows and its ROC-AUC of 0.461 is
*below chance* — so "we beat the baseline" would be a meaningless claim against
it alone. A tuned tree ensemble is the comparison a reviewer will actually ask
for, and it is the one reported as the headline delta.

### Splits

- **Synthetic** uses `dataset.chronological_split`: a single strictly-future cut,
  no shuffling.
- **CIC-IDS2017** uses `dataset.day_aware_split`: chronological *within* each
  capture day, then unioned. This is **not** one strictly-future cut, and the
  distinction matters. Each CIC-IDS2017 day is a largely single-attack-family
  capture, so a global cut would place whole attack families exclusively in test,
  giving the model zero training exposure to them. The cost is that train and test
  interleave across days; windows adjacent to a within-day cut share overlapping
  context, so some temporal leakage is possible. We take that trade deliberately
  and state it rather than describing the split as strictly-future.

`evaluate.py` reports precision/recall/F1/FPR, ROC-AUC and PR-AUC, a threshold
sweep, forecast quality at each rollout horizon, lead-time statistics, a rollout
ablation and a latency profile, and writes `reports/benchmark*.md` plus ROC/PR
curve PNGs.

**Results on real CIC-IDS2017 traffic** (`reports/benchmark_cicids2017.md`,
1,839 held-out windows, 32.3% positive):

| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.311 | 0.783 | 0.445 | 0.829 | 0.452 | 0.413 |
| Random Forest | 0.950 | 0.665 | 0.782 | **0.017** | **0.948** | 0.905 |
| **LSTM World Model** | 0.894 | 0.798 | **0.843** | 0.045 | 0.940 | **0.907** |

The world model wins on F1 by **+0.061** over the strongest baseline, and the
reason is balance rather than raw separability: the random forest is highly
precise but misses 33% of attack windows. In a SOC that is the expensive error.
On pure ranking the two are now level (ROC-AUC 0.940 vs 0.948, PR-AUC 0.907 vs
0.905); an earlier checkpoint trailed the forest clearly on ROC-AUC, and closing
that gap was the main goal of the final-round retrain (§6a).

## 6a. Final-round retrain and model selection

The CIC-IDS2017 checkpoint was retrained once for the final round. Two findings are
worth recording because both are counter-intuitive.

**What helped was training hygiene, not the loss function.** Adding
`ReduceLROnPlateau`, gradient clipping and early stopping, and restoring the
best-validation weights before writing the checkpoint, moved held-out ROC-AUC from
0.898 to 0.940 and rollout retention from 81% to 92%. The previous run had no
scheduler and no early stopping; its `history.json` shows validation loss spiking
from 0.82 at epoch 10 to 2.10 at epoch 14.

**What did not help was the thing that looked most broken.** Measured at the old
weights, the unweighted loss gave the 33-dim dynamics MSE ~70% of the total and the
infiltration head — the quantity every reported metric scores — only 8-12%. Rebalancing
toward the BCE term was the obvious fix and made held-out PR-AUC *worse* (0.839 vs
0.907). A multi-step autoregressive dynamics loss, aimed squarely at rollout drift,
came out level-to-slightly-worse. Both are retained as `--w-dyn` / `--w-inf` /
`--rollout-steps` and default to the configuration that won.

**Selection protocol.** Hyperparameters and seed were chosen on the **validation**
split; test was consulted only to decide promotion, and only after the candidate was
fixed. A candidate replaced an incumbent only if it beat it on *every* held-out
metric. The synthetic checkpoint failed that test — retraining made it worse on all
four — so the original synthetic weights ship unchanged. `tools/eval_seeds.py` and
`tools/ab_compare.py` reproduce both comparisons.

**Split correction.** `day_aware_split` derived per-day window indices as
`rows // window`, while `features.build_state_sequence` buckets on the cumulative row
count. On CIC-IDS2017 that drifted by up to 2 windows and assumed 12,247 windows
against an actual 12,251, putting a small number of windows in the wrong split. The
current checkpoint uses the corrected indices; the previous one did not.

| Metric (identical held-out split) | previous | current |
|---|---|---|
| F1 | 0.832 | **0.843** |
| ROC-AUC | 0.898 | **0.940** |
| PR-AUC | 0.881 | **0.907** |
| 5-step rollout F1 | 0.670 | **0.775** |
| Rollout retention | 81% | **92%** |

### Forecast quality vs horizon — the world-model claim

Every number above comes from a single forward pass over *observed* context,
which is exactly what a plain sequence classifier does. The differentiating
claim is the **autoregressive rollout**, so it is measured separately: at step
*s* the model has fed its own predicted state back in *s−1* times and has seen
no new traffic.

| Horizon | t+1 | t+2 | t+3 | t+4 | t+5 |
|---|---|---|---|---|---|
| F1 | 0.849 | 0.841 | 0.815 | 0.794 | 0.775 |
| ROC-AUC | 0.953 | 0.946 | 0.936 | 0.929 | 0.923 |

Against the same label, the single forward pass scores F1 0.843 and the 5-step
rollout scores 0.775 — **92% retention** while running four of its five steps on
self-generated state. Degradation with horizon is expected and is the signature
of genuine autoregressive dynamics; a model that had only memorised a
current-window mapping would collapse immediately once fed its own output.

### Lead time

Of 40 attack-episode onsets assessable within the held-out period, the model was
already above threshold **before the first malicious flow was recorded** in 18
(45%), median 1 window (200 flows), maximum 3 windows (600 flows). Lead is
measured against the *labelled* onset, so this is warning issued before the
attack traffic exists in the capture — not merely before an analyst noticed.

### CIC-IDS2017 adapter — schema differences & how they were handled

The project ships a working adapter (`netforecast/real_data_adapter.py`) for
**CIC-IDS2017** (the 8 daily CICFlowMeter CSVs — e.g. Kaggle "Network
Intrusion dataset (CIC-IDS-2017)" by chethuhn). This CSV export does
**not** include IP addresses, protocol, wall-clock timestamps, or
packet-level fields (TTL, retransmissions, fragmentation) — only
flow-level CICFlowMeter statistics and a `Label` column. The adapter
therefore:

- Uses row order (CICFlowMeter emits flows in completion order) as a
  chronology proxy, windowing by a fixed **flow count** (200) instead of
  wall-clock seconds — passed as `--window-seconds` downstream.
- Zero-fills the missing packet-level and IP-dependent features (they
  carry no signal here, but the model still learns from all flow-level
  dynamics: byte/packet counts, TCP flags, IAT statistics, ports).
- Maps `Label` values onto the 5 kill-chain stages the problem statement
  asks for: `PortScan`→Reconnaissance, `FTP/SSH-Patator`+`Web Attack *`+
  `Heartbleed`→Initial_Access, `Bot`→Command_And_Control,
  `Infiltration`→Lateral_Movement. **DoS/DDoS rows are dropped** (MITRE
  "Impact", not one of the 5 requested stages). CIC-IDS2017 has **no
  Exfiltration-labelled traffic**, so the model sees zero training
  examples for that stage on this dataset — a real dataset limitation,
  stated here rather than hidden.
- Splits train/val/test **within each day** and unions them
  (`dataset.day_aware_split`), because CIC-IDS2017 dedicates each day to
  one attack family — a single global chronological cut would put entire
  attack types (e.g. all Friday PortScan traffic) only in the test split
  with zero training exposure.

To plug in a different real dataset (CIC-IDS2018, CTU-13, CICIoT2023)
instead, write a similar small adapter following
`real_data_adapter.py`/`simulate_traffic.COLUMNS` as a template: rename
that dataset's columns to the schema in `simulate_traffic.COLUMNS`, derive
`label_stage` from its attack-timeline annotations, and (if it includes
PCAPs) join packet-level fields via **Scapy**/**PyShark** on the flow
5-tuple + time window.

## 7. Deployment & scaling notes

- The whole pipeline runs offline/on-prem: feature extraction, model
  inference, the API and the dashboard make no external network calls,
  meeting the CII (Critical Information Infrastructure) requirement of no
  cloud-API dependency. The dashboard's charting library is **vendored**
  under `server/static/vendor/` rather than loaded from a CDN, so the tool
  renders correctly on an air-gapped network — a CDN `<script>` tag would
  have quietly broken exactly the deployment this is pitched for.

### Measured latency (CPU, single process)

| Operation | Median | p99 |
|---|---|---|
| Featurize 5,000 flows | 18.8 ms | – |
| Single forward pass | 3.01 ms | 4.34 ms |
| 5-step rollout | 9.12 ms | 17.84 ms |
| Rollout + explainability | 9.08 ms | – |

Inference is nowhere near the bottleneck: a window's worth of featurization costs
more than the model does. That is what makes an on-prem CPU-only deployment
realistic, and it is why no GPU appears anywhere in the requirements.

### Service architecture

```
              ┌───────────────────────────────────────────────┐
  CSV upload  │  FastAPI (server/)                            │
  ───────────▶│                                               │
              │  SessionStore ── states, WindowIndex,         │
  Collector   │                  downcast display frame       │
  ───────────▶│  IngestEngine ── rolling buffer per source    │
  POST        │                                               │
  /api/ingest │  REST  /api/session /timeline /forecast       │
              │        /explain /flows /state /mitre          │
              │  WS    /ws/replay/{id}   /ws/live/{src}       │
              └────────────────┬──────────────────────────────┘
                               │  JSON + WebSocket frames
                               ▼
                   SOC dashboard (server/static)
```

Three properties matter for scale:

- **A session never holds the ingested DataFrame.** It keeps the state matrix,
  a flat `WindowIndex` of row offsets, and a downcast display frame of only the
  columns the flow panels render. A 2.45M-row CIC-IDS2017 session costs ~80 MB
  resident instead of ~1.0 GB, and no request deep-copies it.
- **The risk timeline is one batched forward pass**, not one pass per anchor.
- **Replay sleeps in the event loop**, not on the request thread, so the rest of
  the dashboard stays interactive while a capture streams.

- The `POST /api/ingest` endpoint is the seam a production deployment
  attaches a real collector to. It accepts flow batches, maintains a rolling
  per-source buffer, closes a window either on wall clock or on flow count
  (matching whichever the active checkpoint was trained with), scores it and
  pushes the result to subscribed dashboards. `tools/feeder.py` exercises this
  path end-to-end; swapping it for a NetFlow/IPFIX collector, a Zeek `conn.log`
  tail or a Kafka consumer changes nothing downstream.
- For enterprise-scale deployment, `features.py`'s single aggregated
  state vector would be replaced with a per-segment or per-host state and
  the LSTM encoder replaced/augmented with a GNN over a host-interaction
  graph (nodes = hosts, edges = active flows) — the model heads and
  rollout logic in `predict.py` are unchanged by this swap.
- Feature extraction is fully vectorized (single-pass grouped aggregation
  rather than a Python loop over per-window frames): ~280x faster than the
  original implementation, which is what makes windowing a 2.45M-flow capture
  a ~4 s operation instead of a multi-minute one.

## 8. Honesty / scope notes

- The primary trained checkpoint (`models_real/`) is trained on the real
  **CIC-IDS2017** dataset (§6). A second checkpoint (`models/`) trained on
  a synthetic generator built to the CIC-IDS2018 column schema is also
  included, mainly as a controlled testbed during development (it covers
  Exfiltration, which CIC-IDS2017 lacks) and as a template for adapting a
  different real dataset. See §6 for the CIC-IDS2017-specific schema gaps
  (no IPs/timestamps/packet-level fields) and how the adapter handles each.
- The "network state" here is a single aggregated vector for the whole
  monitored segment per time window. A natural extension is a per-host
  graph state with a GNN encoder for larger enterprise topologies (§7).
- K-step rollout accumulates model error autoregressively, as in any
  latent-dynamics/world-model rollout. `evaluate.py` reports metrics at **every**
  horizon step, so the degradation is visible rather than implied: F1 falls from
  0.849 at t+1 to 0.775 at t+5 on real traffic.
- **The shipped real-data checkpoint is the validation-selected seed of three.**
  Held-out PR-AUC across those seeds ranged 0.879-0.907. Its reported numbers are
  reproducible from the committed weights, but a retrain from another seed would
  not necessarily reproduce them.
- **The previously shipped checkpoint was trained against a slightly misaligned
  day-aware split** (see §6a). The drift affected roughly 0.1% of windows; the
  current checkpoint uses the corrected split.
- **On synthetic data the world model does not beat a random forest**
  (F1 0.938 vs 0.941). The generator is close to trivially separable, so the
  baselines saturate and the comparison carries little information. The
  CIC-IDS2017 result is the one that means something. We ship the synthetic
  benchmark anyway rather than quietly dropping an unflattering number.
- **Roughly 9 of the 33 state features are constant zeros on CIC-IDS2017**
  (TTL mean/variance, retransmission rate, fragmentation rate, and the
  IP-derived ratios, since the adapter substitutes placeholder addresses). The
  saliency panel on the real checkpoint therefore ranks a genuinely narrower
  feature set than on the synthetic one, and any claim about, say, TTL variance
  driving a real-data prediction would be false.
- **`window_seconds: 200` on the real checkpoint counts flows, not seconds.**
  CIC-IDS2017 carries no usable wall clock, so the adapter uses row order as the
  chronology proxy. The API exposes `window_unit` on every dataset and session so
  that nothing in the UI or the reports describes a flow count as a duration.
- **The Exfiltration stage has zero training examples on CIC-IDS2017**, and the
  adapter drops DoS/DDoS rows entirely. The 6-way stage head can still emit
  Exfiltration on the real checkpoint; that particular output is not
  evidence-backed and should not be presented as a detection.
- **Best-F1 thresholds reported in the benchmarks are selected on the test
  split.** They are an upper bound on achievable operating-point quality; a real
  deployment would tune the threshold on validation data.
