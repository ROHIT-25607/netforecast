# 2-Minute Demo Video Script

**Setup before recording:** run the reproduce steps in README §4, then
`streamlit run demo/app.py`.

| Time | Screen | Narration |
|---|---|---|
| 0:00–0:15 | Terminal: `python src/simulate_traffic.py`, `train.py`, `evaluate.py` running (sped up / pre-recorded) | "NetForecast ingests flow-level and packet-level traffic features and trains an LSTM world model to learn how network state evolves over time — not just classify individual flows." |
| 0:15–0:35 | Streamlit app, section 1–2 (overview + rolling probability timeline) | "Here's the infiltration-probability timeline across a captured traffic file. Each spike corresponds to a real attack campaign in the data — reconnaissance, lateral movement, and command-and-control beaconing." |
| 0:35–1:00 | Section 3 (K-step forward simulation table) | "At the most recent point in the traffic, NetForecast performs a K-step forward simulation — rolling out its own learned dynamics model — to forecast the probability of infiltration in the *next* several time windows, along with the predicted MITRE ATT&CK stage: reconnaissance, initial access, lateral movement, command-and-control, or exfiltration." |
| 1:00–1:25 | Section 4–5 (attention chart + saliency chart) | "Every prediction is explainable: attention weights show which recent time windows drove the forecast, and feature saliency shows exactly which traffic signals — SYN rates, port-scan ratios, beacon regularity, TTL variance — contributed most." |
| 1:25–1:45 | Section 6 (flagged flows table) | "Analysts can drill straight into the raw flows behind a flagged window for verification." |
| 1:45–2:00 | `reports/benchmark.md` in an editor | "Benchmarked against a logistic-regression baseline on the identical, chronologically held-out test data, the world model shows a measurable improvement — evidence that learning transition dynamics, not just more features, is what drives proactive detection. The entire system runs offline, making it suitable for enterprise and Critical Information Infrastructure deployment." |
