"""
Adapter: CIC-IDS2017 (CICFlowMeter CSVs, "MachineLearningCSV" variant,
e.g. Kaggle "Network Intrusion dataset (CIC-IDS-2017)" by chethuhn) ->
the flow-record schema expected by features.py / simulate_traffic.COLUMNS.

Known limitations of this specific CSV export (stated here for honesty,
also see README "Using the real CIC-IDS2017 dataset"):

  - No source/destination IP columns (anonymised out of the public CSVs).
    IP-dependent state features (n_unique_src, n_unique_dst,
    internal_internal_ratio, external_dst_ratio) therefore carry no real
    signal for this dataset and are left at a constant placeholder.
  - No `Protocol` column in this export -> defaulted to TCP (6), true for
    the overwhelming majority of these captures.
  - No packet-level fields (TTL, retransmissions, fragmentation) -> these
    require the original PCAPs (not included in this CSV set) and are
    zero-filled here. The world model still learns from all flow-level
    dynamics (byte/packet counts, TCP flags, IAT statistics, ports).
  - No wall-clock timestamp column. CICFlowMeter emits rows in the order
    flows complete, which is a reasonable proxy for chronology; we assign
    a synthetic sequential `timestamp` (row index) and window by a fixed
    *number of flows* rather than by seconds. Pass `--window-seconds`
    to train.py/baseline.py as a flow-count window size for this dataset.
  - CIC-IDS2017 has no "Exfiltration"-labelled traffic, so the model will
    see zero training examples for that stage on this dataset alone.
  - DoS/DDoS-labelled rows are dropped: they represent the MITRE "Impact"
    tactic, not one of the 5 kill-chain stages this project forecasts
    (Reconnaissance / Initial Access / Lateral Movement / C2 /
    Exfiltration), so mapping them in would misrepresent the taxonomy.
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd

from simulate_traffic import COLUMNS

# Chronological order of the 8 daily CIC-IDS2017 CSVs (Mon -> Fri).
FILE_ORDER = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# CICFlowMeter column (after .strip()) -> our schema column.
COLUMN_MAP = {
    "Destination Port": "dst_port",
    "Flow Duration": "duration",              # microseconds -> converted to seconds below
    "Total Fwd Packets": "tot_fwd_pkts",
    "Total Backward Packets": "tot_bwd_pkts",
    "Total Length of Fwd Packets": "totlen_fwd_bytes",
    "Total Length of Bwd Packets": "totlen_bwd_bytes",
    "Flow Bytes/s": "flow_byts_s",
    "Flow Packets/s": "flow_pkts_s",
    "Fwd IAT Mean": "fwd_iat_mean",
    "Fwd IAT Std": "fwd_iat_std",
    "Fwd IAT Max": "fwd_iat_max",
    "Bwd IAT Mean": "bwd_iat_mean",
    "Bwd IAT Std": "bwd_iat_std",
    "SYN Flag Count": "syn_flag_cnt",
    "ACK Flag Count": "ack_flag_cnt",
    "FIN Flag Count": "fin_flag_cnt",
    "RST Flag Count": "rst_flag_cnt",
    "PSH Flag Count": "psh_flag_cnt",
    "URG Flag Count": "urg_flag_cnt",
    "Fwd Packet Length Mean": "fwd_pkt_len_mean",
    "Fwd Packet Length Std": "fwd_pkt_len_std",
    "Bwd Packet Length Mean": "bwd_pkt_len_mean",
    "Init_Win_bytes_forward": "init_win_bytes_fwd",
    "Init_Win_bytes_backward": "init_win_bytes_bwd",
    "Label": "raw_label",
}

# raw CIC-IDS2017 label -> our 6-stage taxonomy. Anything not listed here
# (DoS Hulk/GoldenEye/slowloris/Slowhttptest, DDoS) is dropped.
LABEL_MAP = {
    "BENIGN": "Benign",
    "PortScan": "Reconnaissance",
    "FTP-Patator": "Initial_Access",
    "SSH-Patator": "Initial_Access",
    "Heartbleed": "Initial_Access",
    "Bot": "Command_And_Control",
    "Infiltration": "Lateral_Movement",
}


def _map_label(raw: str) -> str:
    raw = raw.strip()
    if raw in LABEL_MAP:
        return LABEL_MAP[raw]
    if raw.startswith("Web Attack"):   # dash character is mangled differently per file/encoding
        return "Initial_Access"
    return None  # DoS/DDoS or anything unrecognised -> drop


def load_and_adapt(raw_dir: str, window_flow_count: int = 200):
    """Returns (adapted_df, day_boundaries) where day_boundaries is the list
    of per-day row counts (post label-filtering), in the same order the
    days were concatenated -- needed downstream to split train/val/test
    *within* each day rather than by pure global chronology (see
    dataset.day_aware_split): CIC-IDS2017 gives each attack family its own
    dedicated day, so a single global 70/15/15 cut would put entire attack
    types (e.g. all of Friday's PortScan/Bot traffic) exclusively in the
    test split with zero training exposure.
    """
    frames = []
    day_boundaries = []
    for fname in FILE_ORDER:
        path = os.path.join(raw_dir, fname)
        if not os.path.exists(path):
            print(f"  [skip] {fname} not found in {raw_dir}")
            continue
        df = pd.read_csv(path, encoding="latin1", low_memory=False)
        df.columns = [c.strip() for c in df.columns]
        df = df.rename(columns=COLUMN_MAP)
        df = df[[c for c in COLUMN_MAP.values() if c in df.columns]]

        df["label_stage"] = df["raw_label"].astype(str).apply(_map_label)
        before = len(df)
        df = df.dropna(subset=["label_stage"]).drop(columns=["raw_label"])
        frames.append(df)
        day_boundaries.append(len(df))
        print(f"  {fname}: {before} -> {len(df)} rows kept "
              f"({df['label_stage'].value_counts().to_dict()})")

    full = pd.concat(frames, ignore_index=True)

    # Fill in fields this CSV export doesn't provide.
    full["protocol"] = 6  # TCP; see module docstring
    full["src_ip"] = "10.0.0.1"        # no real IPs in this export (see docstring)
    full["dst_ip"] = "10.0.0.2"
    full["src_port"] = 0
    full["ttl_mean"] = 0.0
    full["ttl_var"] = 0.0
    full["retransmission_cnt"] = 0.0
    full["frag_flag"] = 0
    full["campaign_id"] = -1

    full["duration"] = pd.to_numeric(full["duration"], errors="coerce").fillna(0) / 1e6  # us -> s

    # Synthetic sequential "timestamp": CICFlowMeter emits rows in completion
    # order, a reasonable chronology proxy. Windowing groups every
    # `window_flow_count` consecutive rows (pass this value as
    # --window-seconds to train.py/baseline.py for this dataset).
    full["timestamp"] = np.arange(len(full))

    for col in COLUMNS:
        if col not in full.columns:
            full[col] = 0
    full = full[COLUMNS]
    full = full.apply(lambda c: c.replace([np.inf, -np.inf], np.nan).fillna(0) if c.dtype != object else c)

    print(f"\nFinal adapted dataset: {len(full)} flows across "
          f"{len(full) // window_flow_count} windows of {window_flow_count} flows each.")
    print(full["label_stage"].value_counts())
    return full, day_boundaries


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--out", default="data/cicids2017_processed.csv")
    ap.add_argument("--window-flow-count", type=int, default=200)
    args = ap.parse_args()

    df, day_boundaries = load_and_adapt(args.raw_dir, args.window_flow_count)
    df.to_csv(args.out, index=False)
    boundaries_path = os.path.splitext(args.out)[0] + ".day_boundaries.json"
    with open(boundaries_path, "w") as f:
        json.dump({"row_counts_per_day": day_boundaries, "files": FILE_ORDER}, f, indent=2)
    print(f"\nWrote {args.out}")
    print(f"Wrote {boundaries_path} (per-day row counts, for day-aware train/val/test split)")
