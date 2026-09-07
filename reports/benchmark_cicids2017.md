# Benchmark: World Model vs Logistic Regression Baseline

Held-out, chronologically-future test set: 1841 windows.

| Model | Precision | Recall | F1 | False Positive Rate |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.313 | 0.780 | 0.447 | 0.828 |
| LSTM World Model (attention, dynamics-trained) | 0.842 | 0.823 | 0.832 | 0.075 |

F1 improvement over baseline: +0.386