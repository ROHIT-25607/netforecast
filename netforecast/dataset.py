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

from .mitre_mapping import ID_TO_STAGE, STAGE_TO_ID, stage_severity

CONTEXT_LEN = 10   # L: number of past windows fed to the model
HORIZON_K = 5      # K: number of future windows for infiltration forecast


class NetworkStateSequenceDataset(Dataset):
    """Supervised sequence examples over a windowed state series.

    ``rollout_steps`` > 1 additionally yields ``future_states`` — the true
    states at ``t+1 … t+R`` — so training can supervise the dynamics head
    autoregressively instead of only one step ahead. Inference rolls K steps,
    so training on one step alone leaves the model exposed to its own drift.
    """

    def __init__(self, states, label_ids, context_len=CONTEXT_LEN, horizon_k=HORIZON_K,
                 rollout_steps=1):
        self.states = states.astype(np.float32)
        self.label_ids = label_ids
        self.L = context_len
        self.K = horizon_k
        self.R = max(1, min(rollout_steps, horizon_k))
        self.n = len(states)
        # A valid anchor t needs L history states ending at t, the K future
        # labels t+1..t+K, and (for the rollout loss) the R future states
        # t+1..t+R. R <= K, so the label bound is always the binding one:
        # t + K <= n - 1  =>  t <= n - K - 1  =>  range stops at n - K.
        self.valid_t = list(range(self.L - 1, self.n - self.K))

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
        item = {
            "x": torch.from_numpy(x),
            "next_state": torch.from_numpy(next_state),
            "next_stage": torch.tensor(next_stage, dtype=torch.long),
            "infiltration": torch.tensor(infiltration, dtype=torch.float32),
            "future_stage_id": torch.tensor(future_stage_id, dtype=torch.long),
        }
        if self.R > 1:
            item["future_states"] = torch.from_numpy(
                self.states[t + 1: t + 1 + self.R])           # (R, F)
        return item


def chronological_split(n, train_frac=0.7, val_frac=0.15):
    """Time-ordered split (no shuffling) so evaluation is on future, unseen time."""
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return slice(0, n_train), slice(n_train, n_train + n_val), slice(n_train + n_val, n)


def day_aware_split(day_row_counts, window_flow_count, train_frac=0.7, val_frac=0.15):
    """Chronological 70/15/15 split applied *within each day* rather than
    once across the whole concatenated sequence, then unioned.

    Needed for datasets (e.g. CIC-IDS2017) where each calendar day is a
    dedicated, largely single-attack-family capture: a single global cut
    would put an entire attack type exclusively in whichever split covers
    that day (usually the test split), giving the model zero training
    exposure to it. Splitting inside each day's own window range instead
    guarantees every attack family present in the data appears in train,
    val AND test, while each day's windows are still used in strict
    forward-time order (no shuffling within a day).

    `day_row_counts`: per-day flow-row counts, in concatenation order
    (as written by real_data_adapter.py's *.day_boundaries.json).
    `window_flow_count`: flows per window (matches --window-seconds used
    to build the state sequence for this dataset).

    Returns three sorted lists of *global* window indices.

    Window indices are derived from the **cumulative** row count, matching how
    `features.build_state_sequence` buckets rows (`timestamp // window`). An
    earlier version accumulated `rows // window` per day instead; because each
    day's row count is not an exact multiple of the window size, that drifted
    from the true index by up to 2 windows on CIC-IDS2017 and assigned a
    handful of windows to the wrong split.
    """
    train_idx, val_idx, test_idx = [], [], []
    cum_rows = 0
    claimed = set()
    for n_rows in day_row_counts:
        first = cum_rows // window_flow_count          # true global index of this day's first window
        cum_rows += n_rows
        last = (cum_rows - 1) // window_flow_count     # ... and of its last

        # A window straddling a day boundary holds rows from both days. The
        # earlier day claims it, so nothing lands in two splits -- and in
        # particular a window carrying the previous day's held-out rows is
        # never pulled into the next day's training set.
        day_windows = [w for w in range(first, last + 1) if w not in claimed]
        claimed.update(day_windows)
        n_windows = len(day_windows)
        if n_windows == 0:
            continue
        n_train = int(n_windows * train_frac)
        n_val = int(n_windows * val_frac)
        train_idx.extend(day_windows[:n_train])
        val_idx.extend(day_windows[n_train:n_train + n_val])
        test_idx.extend(day_windows[n_train + n_val:])

    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


def resolve_data_path(path: str) -> str:
    """Resolve a path recorded in a `config.json`, wherever it is run from.

    Checkpoints written by earlier runs stored paths like
    ``../data/x.json`` -- relative to the old ``src/`` working directory, so
    they break everywhere else. Try the literal path first, then the same name
    relative to the repository root, then the bare filename under ``data/``.
    """
    import os

    if os.path.exists(path):
        return path
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for cand in (os.path.join(root, path.lstrip("./").removeprefix("../")),
                 os.path.join(root, "data", os.path.basename(path))):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        f"{path!r} (recorded in a model config) could not be found, including "
        f"relative to the repository root {root!r}")
