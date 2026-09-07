"""
Trains the LSTM world model with a multi-task loss:

    L = w_dyn * MSE(next_state_pred, next_state_true)
      + w_stage * CrossEntropy(stage_logits, next_stage_true)
      + w_inf * BCE(infiltration_logit, infiltration_label)

Reproducible: fixed seeds, config saved next to the checkpoint.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from dataset import (NetworkStateSequenceDataset, chronological_split, day_aware_split,
                      CONTEXT_LEN, HORIZON_K)
from features import build_state_sequence, normalize_states, N_FEATURES
from mitre_mapping import STAGES
from world_model import WorldModel

SEED = 42


def set_seed(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, opt, device, w_dyn=1.0, w_stage=1.0, w_inf=1.0, train=True):
    model.train(train)
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n = 0
    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["x"].to(device)
            next_state = batch["next_state"].to(device)
            next_stage = batch["next_stage"].to(device)
            infiltration = batch["infiltration"].to(device)

            out = model(x)
            loss = (w_dyn * mse(out["next_state"], next_state)
                    + w_stage * ce(out["stage_logits"], next_stage)
                    + w_inf * bce(out["infiltration_logit"], infiltration))

            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()

            total_loss += loss.item() * x.size(0)
            n += x.size(0)
    return total_loss / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/synthetic_flows.csv")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--window-seconds", type=int, default=30)
    ap.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    ap.add_argument("--horizon-k", type=int, default=HORIZON_K)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--day-boundaries", default=None,
                     help="Path to a *.day_boundaries.json written by real_data_adapter.py. "
                          "When given, train/val/test is split chronologically WITHIN each day "
                          "and unioned, instead of one 70/15/15 cut across the whole file -- "
                          "needed when different days contain different attack families "
                          "(see dataset.day_aware_split).")
    args = ap.parse_args()

    set_seed()
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    states, label_ids, timestamps, _ = build_state_sequence(df, args.window_seconds)
    print(f"Built {len(states)} time windows of {states.shape[1]} features each.")

    if args.day_boundaries:
        with open(args.day_boundaries) as f:
            day_info = json.load(f)
        train_windows, val_windows, test_windows = day_aware_split(
            day_info["row_counts_per_day"], args.window_seconds)
        print(f"Day-aware split from {args.day_boundaries}")
    else:
        tr_sl, va_sl, te_sl = chronological_split(len(states))
        train_windows = list(range(tr_sl.start or 0, tr_sl.stop))
        val_windows = list(range(va_sl.start, va_sl.stop))
        test_windows = list(range(te_sl.start, te_sl.stop))

    # Normalize using statistics from the train split only, to avoid leakage.
    train_states_raw = states[train_windows]
    mean = train_states_raw.mean(axis=0)
    std = train_states_raw.std(axis=0) + 1e-6
    states_norm = (states - mean) / std

    full_ds = NetworkStateSequenceDataset(states_norm, label_ids, args.context_len, args.horizon_k)

    train_set, val_set, test_set = set(train_windows), set(val_windows), set(test_windows)

    def indices_for_windows(window_set):
        return [i for i, t in enumerate(full_ds.valid_t) if t in window_set]

    train_idx = indices_for_windows(train_set)
    val_idx = indices_for_windows(val_set)
    test_idx = indices_for_windows(test_set)
    print(f"Sequence examples -> train:{len(train_idx)} val:{len(val_idx)} test:{len(test_idx)}")

    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=args.batch_size, shuffle=False)

    model = WorldModel(N_FEATURES, len(STAGES), hidden_dim=args.hidden_dim,
                        num_layers=args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    history = []
    for epoch in range(1, args.epochs + 1):
        tr_loss = run_epoch(model, train_loader, opt, device, train=True)
        val_loss = run_epoch(model, val_loader, opt, device, train=False)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss})
        print(f"epoch {epoch:03d}  train_loss={tr_loss:.4f}  val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "world_model.pt"))

    # Save everything needed for reproducible inference.
    config = {
        "n_features": N_FEATURES,
        "n_stages": len(STAGES),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "context_len": args.context_len,
        "horizon_k": args.horizon_k,
        "window_seconds": args.window_seconds,
        "seed": SEED,
        "data": args.data,
        "day_boundaries": args.day_boundaries,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    np.savez(os.path.join(args.out_dir, "norm_stats.npz"), mean=mean, std=std)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # test indices saved so evaluate.py uses the exact same held-out split
    np.savez(os.path.join(args.out_dir, "split_indices.npz"),
              train=np.array(train_idx), val=np.array(val_idx), test=np.array(test_idx))

    print(f"Best val loss: {best_val:.4f}. Model + config saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
