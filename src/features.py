"""
Flow-record -> windowed network-state feature pipeline.

Network state S_t is represented as a fixed-length vector aggregating both
flow-level statistics (volumes, TCP flags, IAT) and packet-derived
statistics (TTL variance, retransmissions, fragmentation) over all flows
observed inside a fixed time window (default 30s). This is what the world
model learns transition dynamics P(S_t+1 | S_t) over.

Works on any DataFrame that has the columns produced by
`simulate_traffic.generate_dataset` (see COLUMNS there). To use a real
CIC-IDS2018 / CTU-13 export, write a small adapter that renames its
columns to this schema (see README "Using a real dataset").
"""
import numpy as np
import pandas as pd

from mitre_mapping import STAGES, STAGE_TO_ID, stage_severity

WINDOW_SECONDS = 30

STATE_FEATURE_NAMES = [
    "n_flows", "n_unique_src", "n_unique_dst", "n_unique_dst_ports",
    "scan_ratio",                      # unique dst ports / n_flows -> scan signature
    "half_open_ratio",                 # SYN with no matching ACK/ FIN -> recon/half-open
    "mean_duration", "std_duration",
    "mean_bytes_fwd", "mean_bytes_bwd", "bytes_ratio_out_in",
    "mean_pkts_fwd", "mean_pkts_bwd",
    "syn_rate", "ack_rate", "fin_rate", "rst_rate", "psh_rate", "urg_rate",
    "mean_fwd_iat", "std_fwd_iat", "max_fwd_iat",
    "beacon_regularity",               # inverse coefficient-of-variation of IAT to same dst -> C2 signature
    "mean_ttl", "mean_ttl_var",
    "mean_win_fwd", "mean_win_bwd",
    "retransmission_rate",
    "frag_rate",
    "internal_internal_ratio",         # lateral-movement signature
    "admin_port_ratio",                # flows to 445/3389/5985/135
    "external_dst_ratio",
    "total_bytes",
]

N_FEATURES = len(STATE_FEATURE_NAMES)
ADMIN_PORTS = {445, 3389, 5985, 135}


def _is_internal(ip: str) -> bool:
    return isinstance(ip, str) and ip.startswith("10.0.")


def compute_window_state(win: pd.DataFrame) -> np.ndarray:
    """Aggregate one window's flow records into a state vector."""
    n = len(win)
    if n == 0:
        return np.zeros(N_FEATURES, dtype=np.float32)

    n_unique_src = win["src_ip"].nunique()
    n_unique_dst = win["dst_ip"].nunique()
    n_unique_ports = win["dst_port"].nunique()
    scan_ratio = n_unique_ports / n

    half_open = ((win["syn_flag_cnt"] > 0) & (win["ack_flag_cnt"] == 0) & (win["fin_flag_cnt"] == 0))
    half_open_ratio = half_open.mean()

    total_fwd = win["totlen_fwd_bytes"].sum()
    total_bwd = win["totlen_bwd_bytes"].sum()
    bytes_ratio = total_fwd / (total_bwd + 1.0)

    internal_internal = (win["src_ip"].apply(_is_internal) & win["dst_ip"].apply(_is_internal)).mean()
    external_dst = (~win["dst_ip"].apply(_is_internal)).mean()
    admin_ratio = win["dst_port"].isin(ADMIN_PORTS).mean()

    # Beacon regularity: for the dst with the most flows, how regular is fwd_iat_mean?
    beacon_reg = 0.0
    if n >= 3:
        top_dst = win["dst_ip"].value_counts().idxmax()
        sub = win.loc[win["dst_ip"] == top_dst, "fwd_iat_mean"]
        if len(sub) >= 3 and sub.mean() > 0:
            cv = sub.std() / (sub.mean() + 1e-6)
            beacon_reg = 1.0 / (1.0 + cv)  # -> 1 for very regular (low CV), -> 0 for bursty/random

    feats = [
        n,
        n_unique_src,
        n_unique_dst,
        n_unique_ports,
        scan_ratio,
        half_open_ratio,
        win["duration"].mean(), win["duration"].std(ddof=0),
        win["totlen_fwd_bytes"].mean(), win["totlen_bwd_bytes"].mean(), bytes_ratio,
        win["tot_fwd_pkts"].mean(), win["tot_bwd_pkts"].mean(),
        win["syn_flag_cnt"].mean(), win["ack_flag_cnt"].mean(), win["fin_flag_cnt"].mean(),
        win["rst_flag_cnt"].mean(), win["psh_flag_cnt"].mean(), win["urg_flag_cnt"].mean(),
        win["fwd_iat_mean"].mean(), win["fwd_iat_std"].mean(), win["fwd_iat_max"].mean(),
        beacon_reg,
        win["ttl_mean"].mean(), win["ttl_var"].mean(),
        win["init_win_bytes_fwd"].mean(), win["init_win_bytes_bwd"].mean(),
        win["retransmission_cnt"].mean(),
        win["frag_flag"].mean(),
        internal_internal,
        admin_ratio,
        external_dst,
        total_fwd + total_bwd,
    ]
    return np.nan_to_num(np.array(feats, dtype=np.float32))


def window_label(win: pd.DataFrame) -> str:
    """Ground-truth stage for a window: the most advanced kill-chain stage present."""
    if len(win) == 0 or "label_stage" not in win:
        return "Benign"
    stages_present = win["label_stage"].unique().tolist()
    return max(stages_present, key=stage_severity)


def build_state_sequence(df: pd.DataFrame, window_seconds: int = WINDOW_SECONDS):
    """Bucket a flow DataFrame into fixed windows and compute (states, labels, timestamps)."""
    df = df.copy()
    df["window_id"] = (df["timestamp"] // window_seconds).astype(int)
    t_min, t_max = df["window_id"].min(), df["window_id"].max()
    all_windows = range(int(t_min), int(t_max) + 1)

    states, labels, timestamps, flow_indices = [], [], [], []
    grouped = {wid: g for wid, g in df.groupby("window_id")}
    for wid in all_windows:
        win = grouped.get(wid, df.iloc[0:0])
        states.append(compute_window_state(win))
        labels.append(window_label(win))
        timestamps.append(wid * window_seconds)
        flow_indices.append(win.index.tolist())

    states = np.stack(states)
    label_ids = np.array([STAGE_TO_ID[s] for s in labels], dtype=np.int64)
    return states, label_ids, np.array(timestamps), flow_indices


def normalize_states(states: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = states.mean(axis=0)
        std = states.std(axis=0) + 1e-6
    return (states - mean) / std, mean, std
