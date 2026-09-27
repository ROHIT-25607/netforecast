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

Implementation note
-------------------
`build_state_sequence` is fully vectorized: every feature is produced by a
single pass of grouped aggregation over the whole frame rather than a
Python loop over per-window sub-DataFrames. Its numerical output is locked
to the original loop implementation by a stored golden fixture (see
tests/test_features.py), which matters because the normalization statistics
baked into the shipped checkpoints were fit on those exact values.
"""
from typing import NamedTuple, Optional

import numpy as np
import pandas as pd

from .mitre_mapping import STAGE_TO_ID, STAGES, stage_severity

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
INTERNAL_PREFIX = "10.0."

# Columns every flow record must carry for the state vector to be computable.
REQUIRED_COLUMNS = (
    "timestamp", "src_ip", "dst_ip", "dst_port",
    "syn_flag_cnt", "ack_flag_cnt", "fin_flag_cnt", "rst_flag_cnt",
    "psh_flag_cnt", "urg_flag_cnt",
    "totlen_fwd_bytes", "totlen_bwd_bytes", "tot_fwd_pkts", "tot_bwd_pkts",
    "duration", "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max",
    "ttl_mean", "ttl_var", "init_win_bytes_fwd", "init_win_bytes_bwd",
    "retransmission_cnt", "frag_flag",
)

# Per-flow columns that become a plain per-window mean.
_MEAN_COLUMNS = (
    "totlen_fwd_bytes", "totlen_bwd_bytes",
    "tot_fwd_pkts", "tot_bwd_pkts",
    "syn_flag_cnt", "ack_flag_cnt", "fin_flag_cnt",
    "rst_flag_cnt", "psh_flag_cnt", "urg_flag_cnt",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max",
    "ttl_mean", "ttl_var",
    "init_win_bytes_fwd", "init_win_bytes_bwd",
    "retransmission_cnt", "frag_flag",
)


class WindowIndex(NamedTuple):
    """Compact, memory-flat replacement for a list-of-lists of row indices.

    Row positions belonging to window ``i`` are ``order[starts[i]:ends[i]]``.
    Holding 2.4M row references in three flat arrays instead of 12k Python
    lists is what lets the API server keep a whole CIC-IDS2017 session
    resident without carrying the DataFrame around.
    """
    order: np.ndarray   # (n_rows,) row positions grouped by window, original order within a window
    starts: np.ndarray  # (n_windows,)
    ends: np.ndarray    # (n_windows,)

    def rows_for(self, window: int, limit: Optional[int] = None) -> np.ndarray:
        s, e = int(self.starts[window]), int(self.ends[window])
        if limit is not None:
            e = min(e, s + limit)
        return self.order[s:e]

    def count_for(self, window: int) -> int:
        return int(self.ends[window] - self.starts[window])

    def __len__(self) -> int:
        return len(self.starts)


def missing_columns(df: pd.DataFrame) -> list:
    """Required schema columns absent from `df` (an empty list means valid)."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _is_internal(ip: str) -> bool:
    return isinstance(ip, str) and ip.startswith(INTERNAL_PREFIX)


def compute_window_state(win: pd.DataFrame) -> np.ndarray:
    """Aggregate one window's flow records into a state vector.

    Reference (loop) implementation. Kept for single-window use by the live
    /api/ingest path, and as the oracle the feature tests check the
    vectorized bulk path against.
    """
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


def _beacon_regularity(w: np.ndarray, dst_codes: np.ndarray, iat: np.ndarray,
                       n_windows: int) -> np.ndarray:
    """Vectorized per-window top-destination IAT regularity.

    For each window: find the destination carrying the most flows, then take
    the inverse coefficient-of-variation (ddof=1, matching Series.std()) of
    that destination's fwd_iat_mean values.
    """
    sub = pd.DataFrame({"w": w, "dst": dst_codes, "iat": iat})
    agg = sub.groupby(["w", "dst"], sort=True)["iat"].agg(["count", "mean", "std"]).reset_index()

    agg = agg.sort_values(["w", "count"], ascending=[True, False], kind="mergesort")
    top = agg.drop_duplicates("w", keep="first").set_index("w")

    # Where two or more destinations tie on the top flow count, the reference
    # implementation's `value_counts().idxmax()` breaks the tie with a
    # non-stable sort, so no ordering rule reproduces it. Those windows are
    # rare (a handful in ~6k), so resolve them by actually calling
    # value_counts() -- cheap, and it keeps the output bit-identical to the
    # values the shipped checkpoints' norm stats were fit on.
    tie_counts = agg[agg["count"].to_numpy() >= 3].groupby("w")["count"].agg(["max", "size"])
    if len(tie_counts):
        cand = agg[agg["count"].to_numpy() >= 3]
        n_at_max = cand[cand["count"].to_numpy() == cand["w"].map(tie_counts["max"]).to_numpy()]
        ambiguous = n_at_max.groupby("w").size()
        ambiguous = ambiguous[ambiguous > 1].index.to_numpy()
    else:
        ambiguous = np.empty(0, dtype=np.int64)

    resolved = {}
    for wi in ambiguous:
        mask = w == wi
        # value_counts on the integer codes picks the same destination as on
        # the original strings: identical counts in identical first-appearance
        # order feed the identical sort.
        chosen = pd.Series(dst_codes[mask]).value_counts().idxmax()
        vals = iat[mask][dst_codes[mask] == chosen]
        resolved[int(wi)] = (len(vals), float(np.mean(vals)),
                             float(np.std(vals, ddof=1)) if len(vals) > 1 else np.nan)

    out = np.zeros(n_windows, dtype=np.float64)
    w_idx = top.index.to_numpy()
    counts_a = top["count"].to_numpy().astype(np.float64)
    means_a = top["mean"].to_numpy()
    stds_a = top["std"].to_numpy()
    for wi, (c, m, s) in resolved.items():
        j = np.searchsorted(w_idx, wi)
        counts_a[j], means_a[j], stds_a[j] = c, m, s

    ok = (counts_a >= 3) & (means_a > 0)
    if ok.any():
        cv = np.nan_to_num(stds_a[ok]) / (means_a[ok] + 1e-6)
        out[w_idx[ok]] = 1.0 / (1.0 + cv)
    return out


def build_state_sequence(df: pd.DataFrame, window_seconds: int = WINDOW_SECONDS,
                         with_index: bool = False):
    """Bucket a flow DataFrame into fixed windows and compute the state sequence.

    Returns ``(states, label_ids, timestamps, window_index)``. `window_index`
    is a :class:`WindowIndex` when ``with_index`` is True and ``None``
    otherwise -- skipping it avoids an argsort over every row.
    """
    missing = missing_columns(df)
    if missing:
        raise KeyError(f"flow records are missing required column(s): {', '.join(missing)}")
    if len(df) == 0:
        raise ValueError("no flow records to window")

    ts = pd.to_numeric(df["timestamp"], errors="coerce").to_numpy(dtype=np.float64)
    if np.isnan(ts).any():
        raise ValueError("`timestamp` column contains non-numeric or missing values")
    wid = (ts // window_seconds).astype(np.int64)

    t_min, t_max = int(wid.min()), int(wid.max())
    n_windows = t_max - t_min + 1
    w = wid - t_min                                   # dense 0-based window index
    counts = np.bincount(w, minlength=n_windows).astype(np.float64)
    nonempty = counts > 0
    safe_counts = np.where(nonempty, counts, 1.0)     # avoid 0-division; zeroed out below

    def _sum(values) -> np.ndarray:
        return np.bincount(w, weights=np.asarray(values, dtype=np.float64), minlength=n_windows)

    def _mean(values) -> np.ndarray:
        return _sum(values) / safe_counts

    # --- whole-frame derived columns, computed once instead of per window ---
    src, dst = df["src_ip"], df["dst_ip"]
    src_int = (src.astype(str).str.startswith(INTERNAL_PREFIX, na=False).to_numpy()
               if src.dtype == object else np.zeros(len(df), dtype=bool))
    dst_int = (dst.astype(str).str.startswith(INTERNAL_PREFIX, na=False).to_numpy()
               if dst.dtype == object else np.zeros(len(df), dtype=bool))

    half_open = ((df["syn_flag_cnt"].to_numpy() > 0)
                 & (df["ack_flag_cnt"].to_numpy() == 0)
                 & (df["fin_flag_cnt"].to_numpy() == 0))
    admin = df["dst_port"].isin(ADMIN_PORTS).to_numpy()

    # --- distinct counts: factorize once, then count uniques per window ---
    src_codes = pd.factorize(src, sort=False)[0]
    dst_codes = pd.factorize(dst, sort=False)[0]
    port_codes = pd.factorize(df["dst_port"], sort=False)[0]

    def _nunique(codes: np.ndarray) -> np.ndarray:
        pairs = pd.MultiIndex.from_arrays([w, codes]).unique()
        return np.bincount(pairs.get_level_values(0).to_numpy(),
                           minlength=n_windows).astype(np.float64)

    n_unique_src = _nunique(src_codes)
    n_unique_dst = _nunique(dst_codes)
    n_unique_ports = _nunique(port_codes)

    # --- aggregates ---
    duration = df["duration"].to_numpy(dtype=np.float64)
    mean_duration = _mean(duration)
    # population std (ddof=0), matching Series.std(ddof=0): sqrt(E[x^2] - E[x]^2)
    std_duration = np.sqrt(np.maximum(_mean(duration ** 2) - mean_duration ** 2, 0.0))

    total_fwd = _sum(df["totlen_fwd_bytes"].to_numpy())
    total_bwd = _sum(df["totlen_bwd_bytes"].to_numpy())

    means = {c: _mean(df[c].to_numpy()) for c in _MEAN_COLUMNS}
    beacon = _beacon_regularity(w, dst_codes,
                                df["fwd_iat_mean"].to_numpy(dtype=np.float64), n_windows)

    states = np.zeros((n_windows, N_FEATURES), dtype=np.float64)
    states[:, 0] = counts
    states[:, 1] = n_unique_src
    states[:, 2] = n_unique_dst
    states[:, 3] = n_unique_ports
    states[:, 4] = n_unique_ports / safe_counts               # scan_ratio
    states[:, 5] = _mean(half_open)                           # half_open_ratio
    states[:, 6] = mean_duration
    states[:, 7] = std_duration
    states[:, 8] = means["totlen_fwd_bytes"]
    states[:, 9] = means["totlen_bwd_bytes"]
    states[:, 10] = total_fwd / (total_bwd + 1.0)             # bytes_ratio_out_in
    states[:, 11] = means["tot_fwd_pkts"]
    states[:, 12] = means["tot_bwd_pkts"]
    states[:, 13] = means["syn_flag_cnt"]
    states[:, 14] = means["ack_flag_cnt"]
    states[:, 15] = means["fin_flag_cnt"]
    states[:, 16] = means["rst_flag_cnt"]
    states[:, 17] = means["psh_flag_cnt"]
    states[:, 18] = means["urg_flag_cnt"]
    states[:, 19] = means["fwd_iat_mean"]
    states[:, 20] = means["fwd_iat_std"]
    states[:, 21] = means["fwd_iat_max"]
    states[:, 22] = beacon
    states[:, 23] = means["ttl_mean"]
    states[:, 24] = means["ttl_var"]
    states[:, 25] = means["init_win_bytes_fwd"]
    states[:, 26] = means["init_win_bytes_bwd"]
    states[:, 27] = means["retransmission_cnt"]
    states[:, 28] = means["frag_flag"]
    states[:, 29] = _mean(src_int & dst_int)                  # internal_internal_ratio
    states[:, 30] = _mean(admin)                              # admin_port_ratio
    states[:, 31] = _mean(~dst_int)                           # external_dst_ratio
    states[:, 32] = total_fwd + total_bwd

    states[~nonempty, :] = 0.0                                # empty window -> zero state
    states = np.nan_to_num(states).astype(np.float32)

    # --- per-window ground truth: the most advanced stage present ---
    label_ids = np.zeros(n_windows, dtype=np.int64)
    if "label_stage" in df.columns:
        stage_codes, uniques = pd.factorize(df["label_stage"], sort=False)
        sev_lut = np.array([stage_severity(str(u)) for u in uniques], dtype=np.int64)
        max_sev = np.zeros(n_windows, dtype=np.int64)
        np.maximum.at(max_sev, w, sev_lut[stage_codes])
        # severity rank == stage id: KILL_CHAIN_ORDER enumerates STAGES in order
        label_ids = np.where(nonempty, max_sev, STAGE_TO_ID["Benign"]).astype(np.int64)

    timestamps = np.arange(t_min, t_max + 1, dtype=np.int64) * window_seconds

    window_index = None
    if with_index:
        order = np.argsort(w, kind="stable")
        ends = np.cumsum(counts).astype(np.int64)
        starts = (ends - counts).astype(np.int64)
        window_index = WindowIndex(order=order, starts=starts, ends=ends)

    return states, label_ids, timestamps, window_index


def normalize_states(states: np.ndarray, mean=None, std=None):
    if mean is None:
        mean = states.mean(axis=0)
        std = states.std(axis=0) + 1e-6
    return (states - mean) / std


__all__ = [
    "STATE_FEATURE_NAMES", "N_FEATURES", "ADMIN_PORTS", "REQUIRED_COLUMNS", "WINDOW_SECONDS",
    "STAGES", "WindowIndex", "missing_columns", "compute_window_state", "window_label",
    "build_state_sequence", "normalize_states",
]
