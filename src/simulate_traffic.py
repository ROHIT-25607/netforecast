"""
Synthetic flow-level + packet-derived traffic generator.

Why synthetic data?
-------------------
CIC-IDS2018 / CTU-13 are multi-GB datasets that cannot be downloaded inside
this environment. This generator produces flow records with the SAME
column schema and semantics as CIC-IDS2018 "CICFlowMeter" CSVs (flow
5-tuple, TCP flag counts, byte/packet counts, IAT statistics) plus
packet-derived fields (TTL variance, window size, retransmissions,
fragmentation) called out in the problem statement, and injects full
kill-chain campaigns (Recon -> Initial Access -> Lateral Movement -> C2 ->
Exfiltration) with realistic per-stage signatures.

`src/dataset.py` / `src/features.py` do not care whether a CSV came from
this generator or from a real CICFlowMeter export as long as the column
schema (see `COLUMNS` below) matches — swapping in a real dataset is a
matter of writing a small column-renaming adapter (see README).
"""
import numpy as np
import pandas as pd

from mitre_mapping import STAGES

RNG_SEED = 42

COLUMNS = [
    "timestamp", "campaign_id", "src_ip", "src_port", "dst_ip", "dst_port",
    "protocol", "duration", "tot_fwd_pkts", "tot_bwd_pkts",
    "totlen_fwd_bytes", "totlen_bwd_bytes", "flow_byts_s", "flow_pkts_s",
    "fwd_iat_mean", "fwd_iat_std", "fwd_iat_max",
    "bwd_iat_mean", "bwd_iat_std",
    "syn_flag_cnt", "ack_flag_cnt", "fin_flag_cnt", "rst_flag_cnt",
    "psh_flag_cnt", "urg_flag_cnt",
    "fwd_pkt_len_mean", "fwd_pkt_len_std", "bwd_pkt_len_mean",
    "init_win_bytes_fwd", "init_win_bytes_bwd",
    "ttl_mean", "ttl_var", "retransmission_cnt", "frag_flag",
    "label_stage",
]

INTERNAL_SUBNET = "10.0.0."
DMZ_SUBNET = "10.0.1."


def _rand_ip(rng, subnet=None):
    if subnet is None:
        return f"203.0.{rng.integers(0, 255)}.{rng.integers(1, 255)}"  # external
    return f"{subnet}{rng.integers(2, 254)}"


def _benign_flow(rng, t):
    proto = rng.choice([6, 17], p=[0.8, 0.2])  # TCP / UDP
    dst_port = rng.choice([80, 443, 443, 443, 53, 22, 3306, 8080])
    dur = rng.exponential(2.0) + 0.01
    fwd_pkts = rng.integers(2, 40)
    bwd_pkts = rng.integers(2, 40)
    return {
        "timestamp": t,
        "campaign_id": -1,
        "src_ip": _rand_ip(rng, INTERNAL_SUBNET),
        "src_port": rng.integers(1024, 65535),
        "dst_ip": _rand_ip(rng) if rng.random() < 0.6 else _rand_ip(rng, INTERNAL_SUBNET),
        "dst_port": dst_port,
        "protocol": proto,
        "duration": dur,
        "tot_fwd_pkts": fwd_pkts,
        "tot_bwd_pkts": bwd_pkts,
        "totlen_fwd_bytes": fwd_pkts * max(rng.normal(500, 150), 40),
        "totlen_bwd_bytes": bwd_pkts * max(rng.normal(500, 150), 40),
        "flow_byts_s": max(rng.normal(4000, 1500), 10),
        "flow_pkts_s": max(rng.normal(20, 8), 0.5),
        "fwd_iat_mean": rng.exponential(0.3),
        "fwd_iat_std": rng.exponential(0.1),
        "fwd_iat_max": rng.exponential(0.6),
        "bwd_iat_mean": rng.exponential(0.3),
        "bwd_iat_std": rng.exponential(0.1),
        "syn_flag_cnt": 1,
        "ack_flag_cnt": fwd_pkts + bwd_pkts - 2,
        "fin_flag_cnt": rng.choice([0, 1], p=[0.3, 0.7]),
        "rst_flag_cnt": 0,
        "psh_flag_cnt": rng.integers(0, 5),
        "urg_flag_cnt": 0,
        "fwd_pkt_len_mean": max(rng.normal(500, 150), 40),
        "fwd_pkt_len_std": max(rng.normal(80, 20), 1),
        "bwd_pkt_len_mean": max(rng.normal(500, 150), 40),
        "init_win_bytes_fwd": rng.choice([8192, 14600, 65535]),
        "init_win_bytes_bwd": rng.choice([8192, 14600, 65535]),
        "ttl_mean": rng.choice([64, 128, 255]) - rng.integers(0, 3),
        "ttl_var": rng.exponential(0.2),
        "retransmission_cnt": rng.poisson(0.1),
        "frag_flag": 0,
        "label_stage": "Benign",
    }


def _recon_flow(rng, t, campaign_id, attacker_ip, target_subnet, dst_port):
    """Port scan: short, half-open (SYN only / SYN+RST), sequential/random ports."""
    return {
        "timestamp": t, "campaign_id": campaign_id,
        "src_ip": attacker_ip, "src_port": rng.integers(1024, 65535),
        "dst_ip": _rand_ip(rng, target_subnet), "dst_port": dst_port,
        "protocol": 6,
        "duration": rng.exponential(0.02) + 0.001,
        "tot_fwd_pkts": 1, "tot_bwd_pkts": rng.choice([0, 1], p=[0.7, 0.3]),
        "totlen_fwd_bytes": 40 + rng.integers(0, 20), "totlen_bwd_bytes": 0,
        "flow_byts_s": max(rng.normal(500, 200), 10),
        "flow_pkts_s": max(rng.normal(150, 40), 10),
        "fwd_iat_mean": rng.exponential(0.01), "fwd_iat_std": rng.exponential(0.005),
        "fwd_iat_max": rng.exponential(0.02),
        "bwd_iat_mean": 0.0, "bwd_iat_std": 0.0,
        "syn_flag_cnt": 1, "ack_flag_cnt": 0, "fin_flag_cnt": 0,
        "rst_flag_cnt": rng.choice([0, 1], p=[0.5, 0.5]),
        "psh_flag_cnt": 0, "urg_flag_cnt": 0,
        "fwd_pkt_len_mean": 40, "fwd_pkt_len_std": 2, "bwd_pkt_len_mean": 0,
        "init_win_bytes_fwd": rng.choice([1024, 512, 65535]),
        "init_win_bytes_bwd": 0,
        "ttl_mean": rng.choice([64, 128]) , "ttl_var": 0.01,
        "retransmission_cnt": 0, "frag_flag": 0,
        "label_stage": "Reconnaissance",
    }


def _initial_access_flow(rng, t, campaign_id, attacker_ip, victim_ip, dst_port):
    """Exploit attempt against exposed service: full handshake, payload spike."""
    fwd_pkts = rng.integers(10, 60)
    return {
        "timestamp": t, "campaign_id": campaign_id,
        "src_ip": attacker_ip, "src_port": rng.integers(1024, 65535),
        "dst_ip": victim_ip, "dst_port": dst_port, "protocol": 6,
        "duration": rng.exponential(1.5) + 0.05,
        "tot_fwd_pkts": fwd_pkts, "tot_bwd_pkts": rng.integers(5, 30),
        "totlen_fwd_bytes": fwd_pkts * max(rng.normal(1200, 400), 60),
        "totlen_bwd_bytes": max(rng.normal(3000, 1000), 0),
        "flow_byts_s": max(rng.normal(9000, 3000), 10),
        "flow_pkts_s": max(rng.normal(40, 15), 1),
        "fwd_iat_mean": rng.exponential(0.05), "fwd_iat_std": rng.exponential(0.02),
        "fwd_iat_max": rng.exponential(0.1),
        "bwd_iat_mean": rng.exponential(0.08), "bwd_iat_std": rng.exponential(0.03),
        "syn_flag_cnt": 1, "ack_flag_cnt": fwd_pkts, "fin_flag_cnt": 1,
        "rst_flag_cnt": 0, "psh_flag_cnt": rng.integers(3, 10), "urg_flag_cnt": 0,
        "fwd_pkt_len_mean": max(rng.normal(1200, 300), 60),
        "fwd_pkt_len_std": max(rng.normal(300, 80), 1),
        "bwd_pkt_len_mean": max(rng.normal(600, 200), 40),
        "init_win_bytes_fwd": 65535, "init_win_bytes_bwd": 65535,
        "ttl_mean": rng.choice([64, 128]), "ttl_var": 0.05,
        "retransmission_cnt": rng.poisson(1.0), "frag_flag": rng.choice([0, 1], p=[0.8, 0.2]),
        "label_stage": "Initial_Access",
    }


def _lateral_movement_flow(rng, t, campaign_id, foothold_ip, target_subnet):
    """Internal-to-internal admin-protocol traffic (SMB/RDP/WinRM)."""
    dst_port = rng.choice([445, 3389, 5985, 135])
    return {
        "timestamp": t, "campaign_id": campaign_id,
        "src_ip": foothold_ip, "src_port": rng.integers(1024, 65535),
        "dst_ip": _rand_ip(rng, target_subnet), "dst_port": dst_port, "protocol": 6,
        "duration": rng.exponential(0.8) + 0.02,
        "tot_fwd_pkts": rng.integers(5, 25), "tot_bwd_pkts": rng.integers(5, 25),
        "totlen_fwd_bytes": max(rng.normal(2000, 800), 60),
        "totlen_bwd_bytes": max(rng.normal(2000, 800), 60),
        "flow_byts_s": max(rng.normal(5000, 2000), 10),
        "flow_pkts_s": max(rng.normal(30, 10), 1),
        "fwd_iat_mean": rng.exponential(0.1), "fwd_iat_std": rng.exponential(0.05),
        "fwd_iat_max": rng.exponential(0.2),
        "bwd_iat_mean": rng.exponential(0.1), "bwd_iat_std": rng.exponential(0.05),
        "syn_flag_cnt": 1, "ack_flag_cnt": rng.integers(4, 20),
        "fin_flag_cnt": rng.choice([0, 1]), "rst_flag_cnt": rng.choice([0, 1], p=[0.85, 0.15]),
        "psh_flag_cnt": rng.integers(1, 6), "urg_flag_cnt": 0,
        "fwd_pkt_len_mean": max(rng.normal(400, 150), 40),
        "fwd_pkt_len_std": max(rng.normal(100, 30), 1),
        "bwd_pkt_len_mean": max(rng.normal(400, 150), 40),
        "init_win_bytes_fwd": 65535, "init_win_bytes_bwd": 65535,
        "ttl_mean": 64, "ttl_var": 0.01,
        "retransmission_cnt": rng.poisson(0.3), "frag_flag": 0,
        "label_stage": "Lateral_Movement",
    }


def _c2_flow(rng, t, campaign_id, infected_ip, c2_ip, beacon_period):
    """Low-volume periodic beaconing to an external controller."""
    jitter = rng.normal(0, beacon_period * 0.05)
    return {
        "timestamp": t, "campaign_id": campaign_id,
        "src_ip": infected_ip, "src_port": rng.integers(1024, 65535),
        "dst_ip": c2_ip, "dst_port": rng.choice([443, 8443, 53]), "protocol": 6,
        "duration": rng.exponential(0.3) + 0.01,
        "tot_fwd_pkts": rng.integers(1, 4), "tot_bwd_pkts": rng.integers(1, 4),
        "totlen_fwd_bytes": max(rng.normal(150, 40), 20),
        "totlen_bwd_bytes": max(rng.normal(150, 40), 20),
        "flow_byts_s": max(rng.normal(300, 100), 5),
        "flow_pkts_s": max(rng.normal(5, 2), 0.5),
        "fwd_iat_mean": abs(beacon_period + jitter), "fwd_iat_std": abs(jitter) + 0.01,
        "fwd_iat_max": abs(beacon_period + jitter) * 1.1,
        "bwd_iat_mean": abs(beacon_period + jitter), "bwd_iat_std": abs(jitter) + 0.01,
        "syn_flag_cnt": 1, "ack_flag_cnt": 2, "fin_flag_cnt": 1, "rst_flag_cnt": 0,
        "psh_flag_cnt": 1, "urg_flag_cnt": 0,
        "fwd_pkt_len_mean": max(rng.normal(150, 40), 20),
        "fwd_pkt_len_std": max(rng.normal(10, 3), 1),
        "bwd_pkt_len_mean": max(rng.normal(150, 40), 20),
        "init_win_bytes_fwd": 65535, "init_win_bytes_bwd": 65535,
        "ttl_mean": rng.choice([54, 118, 245]), "ttl_var": 0.02,
        "retransmission_cnt": 0, "frag_flag": 0,
        "label_stage": "Command_And_Control",
    }


def _exfil_flow(rng, t, campaign_id, infected_ip, c2_ip):
    """Sustained, high-volume, mostly-outbound transfer."""
    fwd_pkts = rng.integers(500, 3000)
    bwd_pkts = rng.integers(10, 100)
    return {
        "timestamp": t, "campaign_id": campaign_id,
        "src_ip": infected_ip, "src_port": rng.integers(1024, 65535),
        "dst_ip": c2_ip, "dst_port": rng.choice([443, 8443]), "protocol": 6,
        "duration": rng.uniform(20, 120),
        "tot_fwd_pkts": fwd_pkts, "tot_bwd_pkts": bwd_pkts,
        "totlen_fwd_bytes": fwd_pkts * max(rng.normal(1400, 100), 200),
        "totlen_bwd_bytes": bwd_pkts * max(rng.normal(200, 50), 20),
        "flow_byts_s": max(rng.normal(60000, 20000), 100),
        "flow_pkts_s": max(rng.normal(80, 20), 5),
        "fwd_iat_mean": rng.exponential(0.01), "fwd_iat_std": rng.exponential(0.005),
        "fwd_iat_max": rng.exponential(0.05),
        "bwd_iat_mean": rng.exponential(0.3), "bwd_iat_std": rng.exponential(0.1),
        "syn_flag_cnt": 1, "ack_flag_cnt": fwd_pkts, "fin_flag_cnt": 1, "rst_flag_cnt": 0,
        "psh_flag_cnt": rng.integers(50, 200), "urg_flag_cnt": 0,
        "fwd_pkt_len_mean": max(rng.normal(1400, 100), 200),
        "fwd_pkt_len_std": max(rng.normal(50, 15), 1),
        "bwd_pkt_len_mean": max(rng.normal(200, 50), 20),
        "init_win_bytes_fwd": 65535, "init_win_bytes_bwd": 65535,
        "ttl_mean": rng.choice([54, 118, 245]), "ttl_var": 0.02,
        "retransmission_cnt": rng.poisson(2.0), "frag_flag": rng.choice([0, 1], p=[0.6, 0.4]),
        "label_stage": "Exfiltration",
    }


def _generate_campaign(rng, campaign_id, t0):
    """Build the flow list for one full kill-chain campaign starting at t0 (seconds)."""
    attacker_ip = _rand_ip(rng)
    target_subnet = INTERNAL_SUBNET
    victim_ip = _rand_ip(rng, target_subnet)
    c2_ip = _rand_ip(rng)
    flows = []
    t = t0

    # 1. Reconnaissance: scan 40-150 ports over 30-120s (evades naive thresholds via slow rate)
    n_scan = rng.integers(40, 150)
    scan_ports = rng.choice(np.arange(1, 65535), size=n_scan, replace=False)
    scan_window = rng.uniform(30, 120)
    for i, p in enumerate(scan_ports):
        t += scan_window / n_scan * rng.uniform(0.5, 1.5)
        flows.append(_recon_flow(rng, t, campaign_id, attacker_ip, target_subnet, int(p)))

    # 2. Initial Access: exploit exposed service shortly after recon
    t += rng.uniform(60, 600)
    exposed_port = int(rng.choice([21, 22, 80, 443, 445, 3389]))
    flows.append(_initial_access_flow(rng, t, campaign_id, attacker_ip, victim_ip, exposed_port))

    # 3. Lateral Movement: pivot from victim to several internal hosts
    t += rng.uniform(120, 900)
    n_lateral = rng.integers(5, 20)
    for _ in range(n_lateral):
        t += rng.uniform(5, 60)
        flows.append(_lateral_movement_flow(rng, t, campaign_id, victim_ip, target_subnet))

    # 4. Command & Control: periodic beaconing over an extended dwell period
    t += rng.uniform(60, 300)
    beacon_period = rng.uniform(20, 90)
    n_beacons = rng.integers(30, 100)
    for _ in range(n_beacons):
        t += beacon_period
        flows.append(_c2_flow(rng, t, campaign_id, victim_ip, c2_ip, beacon_period))

    # 5. Exfiltration: bulk transfer near the end of dwell time
    t += rng.uniform(30, 180)
    n_exfil = rng.integers(1, 4)
    for _ in range(n_exfil):
        t += rng.uniform(5, 30)
        flows.append(_exfil_flow(rng, t, campaign_id, victim_ip, c2_ip))

    return flows


def generate_dataset(n_benign=20000, n_campaigns=15, duration_s=48 * 3600, seed=RNG_SEED):
    """Generate an interleaved benign+attack flow-record dataset.

    Returns a DataFrame sorted by timestamp with the CIC-IDS2018-style
    schema defined in COLUMNS.
    """
    rng = np.random.default_rng(seed)
    rows = []

    # Benign background traffic spread uniformly across the whole window.
    benign_times = np.sort(rng.uniform(0, duration_s, size=n_benign))
    for t in benign_times:
        rows.append(_benign_flow(rng, float(t)))

    # Attack campaigns start at random points across the whole window (including
    # near the end, so held-out "future" test splits still contain positive
    # examples -- a campaign starting late may simply be cut off mid-kill-chain,
    # which is realistic: an ongoing attack observed at the end of the capture).
    campaign_starts = rng.uniform(0, duration_s * 0.95, size=n_campaigns)
    for cid, t0 in enumerate(sorted(campaign_starts)):
        rows.extend(_generate_campaign(rng, cid, float(t0)))

    df = pd.DataFrame(rows, columns=COLUMNS)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic_flows.csv")
    ap.add_argument("--n-benign", type=int, default=20000)
    ap.add_argument("--n-campaigns", type=int, default=15)
    ap.add_argument("--duration-hours", type=float, default=48)
    ap.add_argument("--seed", type=int, default=RNG_SEED)
    args = ap.parse_args()

    df = generate_dataset(
        n_benign=args.n_benign,
        n_campaigns=args.n_campaigns,
        duration_s=args.duration_hours * 3600,
        seed=args.seed,
    )
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} flow records to {args.out}")
    print(df["label_stage"].value_counts())
