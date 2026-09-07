"""
Mapping between internal attack-stage labels and MITRE ATT&CK tactics.

The dataset generator (and any real dataset loader) is expected to label
flows/windows with one of the STAGES below. This module is the single
source of truth for stage <-> MITRE ATT&CK tactic mapping used across
training, inference and the demo UI.
"""

STAGES = [
    "Benign",
    "Reconnaissance",
    "Initial_Access",
    "Lateral_Movement",
    "Command_And_Control",
    "Exfiltration",
]

STAGE_TO_ID = {s: i for i, s in enumerate(STAGES)}
ID_TO_STAGE = {i: s for i, s in enumerate(STAGES)}

# MITRE ATT&CK (Enterprise) tactic reference for each stage.
# https://attack.mitre.org/tactics/enterprise/
MITRE_TACTIC = {
    "Benign": {
        "tactic_id": "-",
        "tactic_name": "N/A",
        "description": "No malicious activity observed.",
    },
    "Reconnaissance": {
        "tactic_id": "TA0043",
        "tactic_name": "Reconnaissance",
        "description": "Active scanning of ports/hosts to discover victim network "
                        "topology and running services (e.g. T1595 Active Scanning).",
    },
    "Initial_Access": {
        "tactic_id": "TA0001",
        "tactic_name": "Initial Access",
        "description": "Attempted or successful exploitation of a public-facing or "
                        "exposed service to gain a foothold (e.g. T1190 Exploit Public-Facing App).",
    },
    "Lateral_Movement": {
        "tactic_id": "TA0008",
        "tactic_name": "Lateral Movement",
        "description": "Use of valid/compromised credentials or admin protocols "
                        "(SMB/RDP/WinRM) to pivot across internal hosts (e.g. T1021).",
    },
    "Command_And_Control": {
        "tactic_id": "TA0011",
        "tactic_name": "Command and Control",
        "description": "Periodic beaconing to an external controller to receive "
                        "instructions (e.g. T1071 Application Layer Protocol).",
    },
    "Exfiltration": {
        "tactic_id": "TA0010",
        "tactic_name": "Exfiltration",
        "description": "Sustained, high-volume outbound transfer of staged data "
                        "(e.g. T1041 Exfiltration Over C2 Channel).",
    },
}

# Severity ordering used to decide "how far along the kill chain" a window is.
KILL_CHAIN_ORDER = {s: i for i, s in enumerate(STAGES)}


def stage_severity(stage: str) -> int:
    return KILL_CHAIN_ORDER.get(stage, 0)
