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

## 6. Benchmark methodology

`baseline.py` trains a logistic regression on the *identical* flattened
L-step context window and K-step infiltration label used by the world
model, so the comparison isolates the value of the learned recurrent
dynamics rather than differences in features or horizon. Both models are
evaluated on the same chronologically-held-out, strictly-future test
split (`dataset.chronological_split`, cached in
`models/split_indices.npz`) — no shuffling, so the test period is truly
unseen. `evaluate.py` reports precision, recall, F1 and false-positive
rate for both and writes `reports/benchmark.md`.

## 7. Deployment & scaling notes

- The whole pipeline runs offline/on-prem: feature extraction, model
  inference and the Streamlit UI have no external network calls, meeting
  the CII (Critical Information Infrastructure) requirement of no
  cloud-API dependency.
- For enterprise-scale deployment, `features.py`'s single aggregated
  state vector would be replaced with a per-segment or per-host state and
  the LSTM encoder replaced/augmented with a GNN over a host-interaction
  graph (nodes = hosts, edges = active flows) — the model heads and
  rollout logic in `predict.py` are unchanged by this swap.
- Ingest can be extended from CSV batch files to a streaming feature
  pipeline (e.g. windowed aggregation over a Kafka/NetFlow collector feed)
  without touching the model.

## 8. Honesty / scope notes

Training uses a synthetic flow generator built to the CIC-IDS2018 column
schema and known kill-chain signatures (real datasets are multi-GB and
could not be fetched in this build environment). See README §6 for the
exact adapter steps to point the identical pipeline at a real
CIC-IDS2018/CTU-13 export.
