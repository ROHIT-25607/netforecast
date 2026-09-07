"""
Turns a (states, labels) time series into supervised sequence-model
training examples:

  input:  S_{t-L+1 .. t}            (L past states, "context window")
  targets:
      next_state:   S_{t+1}                          (dynamics regression)
      next_stage:   stage(t+1)                        (classification)
      infiltration: max severity in (t+1 .. t+K) > 0   (binary, K-step horizon)
      future_stage_id: worst stage id in (t+1..t+K)    (for MITRE stage forecast)
"""
import numpy as np
import torch
from torch.utils.data import Dataset

from mitre_mapping import STAGE_TO_ID, stage_severity, ID_TO_STAGE

CONTEXT_LEN = 10   # L: number of past windows fed to the model
HORIZON_K = 5      # K: number of future windows for infiltration forecast


class NetworkStateSequenceDataset(Dataset):
    def __init__(self, states, label_ids, context_len=CONTEXT_LEN, horizon_k=HORIZON_K):
        self.states = states.astype(np.float32)
        self.label_ids = label_ids
        self.L = context_len
        self.K = horizon_k
        self.n = len(states)
        # valid anchor t: needs L history states ending at t, and K future for label
        self.valid_t = list(range(self.L - 1, self.n - self.K - 1))

    def __len__(self):
        return len(self.valid_t)

    def __getitem__(self, idx):
        t = self.valid_t[idx]
        x = self.states[t - self.L + 1: t + 1]                 # (L, F)
        next_state = self.states[t + 1]                        # (F,)
        next_stage = self.label_ids[t + 1]
        future = self.label_ids[t + 1: t + 1 + self.K]
        infiltration = int(max(future) > STAGE_TO_ID["Benign"])
        future_stage_id = int(max(future, key=lambda s: stage_severity(ID_TO_STAGE[s])))
        return {
            "x": torch.from_numpy(x),
            "next_state": torch.from_numpy(next_state),
            "next_stage": torch.tensor(next_stage, dtype=torch.long),
            "infiltration": torch.tensor(infiltration, dtype=torch.float32),
            "future_stage_id": torch.tensor(future_stage_id, dtype=torch.long),
        }


def chronological_split(n, train_frac=0.7, val_frac=0.15):
    """Time-ordered split (no shuffling) so evaluation is on future, unseen time."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)
