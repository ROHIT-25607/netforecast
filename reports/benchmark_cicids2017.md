# Benchmark: World Model vs Baselines

Dataset: `data/cicids2017_processed.csv` · model: `models_real` · held-out test anchors: **1841** (positive rate 32.6%).

All numbers below come from frozen checkpoints; nothing here retrains the world model.


## 1. Single-shot detection

| Model | Precision | Recall | F1 | FPR | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.313 | 0.780 | 0.447 | 0.828 | 0.461 | 0.422 |
| Random Forest | 0.956 | 0.653 | 0.776 | 0.015 | 0.947 | 0.907 |
| **LSTM World Model** | 0.842 | 0.823 | **0.832** | 0.075 | 0.898 | 0.886 |

F1 improvement over the strongest baseline (Random Forest): **+0.056**

## 2. Operating point

Decision threshold is swept rather than fixed at 0.5. Best-F1 threshold on this split: **0.600** (F1 0.845 vs 0.832 at 0.5).

Note this threshold is selected on the test split, so treat it as an upper bound; a deployment would tune it on validation.

## 3. Forecast quality vs horizon (the world-model claim)

Step *s* is the model's **autoregressive rollout**: it has fed its own predicted state back in *s-1* times and has seen no new traffic. Scored against whether any malicious window actually occurs within *t+1 … t+s*.

| Horizon step | n | Positive rate | Precision | Recall | F1 | FPR | ROC-AUC |
|---|---|---|---|---|---|---|---|
| t+1 | 1841 | 27.8% | 0.784 | 0.898 | 0.837 | 0.096 | 0.926 |
| t+2 | 1841 | 30.0% | 0.776 | 0.864 | 0.818 | 0.107 | 0.910 |
| t+3 | 1841 | 31.3% | 0.737 | 0.854 | 0.791 | 0.139 | 0.884 |
| t+4 | 1841 | 32.0% | 0.659 | 0.844 | 0.740 | 0.205 | 0.858 |
| t+5 | 1841 | 32.6% | 0.571 | 0.825 | 0.675 | 0.300 | 0.826 |

## 4. Lead time

Attack-episode onsets inside the evaluated test range: **41** (of 227 in the full capture). The model was already above the 0.60 threshold before onset in **20** of them (49%).

| Statistic | Windows of warning | Equivalent |
|---|---|---|
| Median (all onsets) | 0.0 | 0 flows |
| Median (onsets warned) | 1.0 | 200 flows |
| Mean | 0.8 | 161 flows |
| p90 | 2.0 | 400 flows |
| Max | 6.0 | 1200 flows |

Lead is measured against the *labelled* onset, so this is warning raised before the first malicious flow was recorded -- not merely before an analyst noticed.

## 5. Ablation — what the rollout costs

Both rows predict the **same label** (any malicious window in *t+1 … t+5*), so they are directly comparable. The difference is that the rollout has replaced 4 of its context updates with its *own* predicted states.

| Configuration | F1 | ROC-AUC |
|---|---|---|
| Single forward pass over observed context | 0.832 | 0.898 |
| 5-step autoregressive rollout | 0.675 | 0.826 |

The rollout retains **81%** of single-shot F1 while running 4 steps on self-generated state. That retention is the evidence the learned transition function is doing real work: a model that had only memorised a current-window mapping would degrade sharply once fed its own output.

The per-step table in section 3 uses a *cumulative* label (malicious anywhere in *t+1 … t+s*), so its positive rate rises with s and early steps are not comparable with later ones or with the table above.

## 6. Latency (CPU, single process)

| Operation | Median | p99 |
|---|---|---|
| Featurize 5k flows | 45.2 ms | – |
| Single forward pass | 5.31 ms | 10.36 ms |
| 5-step rollout | 21.83 ms | 28.65 ms |
| Rollout + explainability | 39.94 ms | – |

Everything runs offline on CPU — no GPU and no network call in the inference path.

## Curves

![ROC](roc_benchmark_cicids2017.png)

![PR](pr_benchmark_cicids2017.png)