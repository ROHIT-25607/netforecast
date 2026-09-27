"""NetForecast — an AI world model for network attack forecasting.

Learns the transition dynamics P(S_t+1 | S_t) over windowed network-traffic
state and forecasts attacker progression K steps ahead, mapped to MITRE
ATT&CK tactics. Runs fully offline on CPU.
"""

__version__ = "1.0.0"
