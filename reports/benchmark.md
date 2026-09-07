# Benchmark: World Model vs Logistic Regression Baseline

Held-out, chronologically-future test set: 859 windows.

| Model | Precision | Recall | F1 | False Positive Rate |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.962 | 0.901 | 0.931 | 0.026 |
| LSTM World Model (attention, dynamics-trained) | 0.930 | 0.945 | 0.938 | 0.053 |

F1 improvement over baseline: +0.007