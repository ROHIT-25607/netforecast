"""
Trains the LSTM world model with a multi-task loss:

    L = w_dyn   * MSE over an R-step autoregressive dynamics rollout
      + w_stage * CrossEntropy(stage_logits, next_stage)
      + w_inf   * BCE(infiltration_logit, infiltration_label)

Two things here matter more than they look.

**Loss weighting.** The dynamics term regresses 33 dimensions while the
infiltration term is a single logit, so an unweighted sum lets the regression
dominate: measured on the previously shipped checkpoints, MSE accounted for
~70% of the total loss and the infiltration head — the one every reported
metric scores — for 8-12%. The defaults below correct that balance.

**Multi-step dynamics (`--rollout-steps`).** Inference rolls the model K steps
autoregressively, feeding its own predicted state back in. Training on a single
step leaves it exposed to its own drift, which is exactly the degradation seen
across the horizon table. Supervising the rollout against the true future
states attacks that at the cause. `--rollout-steps 1` reproduces the original
one-step behaviour.

Reproducible: seeds fixed across python/numpy/torch, config saved next to the
checkpoint, best-validation weights restored before anything is written.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from .dataset import (
    CONTEXT_LEN,
    HORIZON_K,
    NetworkStateSequenceDataset,
    chronological_split,
    day_aware_split,
    resolve_data_path,
)
from .features import N_FEATURES, build_state_sequence
from .mitre_mapping import STAGES
from .world_model import WorldModel

SEED = 42


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def rollout_dynamics_loss(model, x, future_states, mse):
    """Autoregressive R-step dynamics loss.

    Step 0 is the ordinary one-step prediction; each subsequent step consumes
    the model's own output, so the gradient teaches the predicted states to
    stay on the data manifold when fed back at inference time.
    """
    R = future_states.shape[1]
    cur = x
    total = 0.0
    for r in range(R):
        out = model(cur)
        pred = out["next_state"]
        total = total + mse(pred, future_states[:, r])
        if r < R - 1:
            # No detach: the gradient must flow back through the whole rollout,
            # otherwise later steps cannot teach earlier ones to stay stable.
            cur = torch.cat([cur[:, 1:], pred.unsqueeze(1)], dim=1)
    return total / R


def run_epoch(model, loader, opt, device, w_dyn, w_stage, w_inf,
              train=True, clip=1.0, rollout_steps=1):
    model.train(train)
    mse = torch.nn.MSELoss()
    ce = torch.nn.CrossEntropyLoss()
    bce = torch.nn.BCEWithLogitsLoss()
    total_loss, parts, n = 0.0, {"dyn": 0.0, "stage": 0.0, "inf": 0.0}, 0

    with torch.set_grad_enabled(train):
        for batch in loader:
            x = batch["x"].to(device)
            next_stage = batch["next_stage"].to(device)
            infiltration = batch["infiltration"].to(device)

            out = model(x)
            if rollout_steps > 1 and "future_states" in batch:
                dyn = rollout_dynamics_loss(model, x, batch["future_states"].to(device), mse)
            else:
                dyn = mse(out["next_state"], batch["next_state"].to(device))

            l_stage = ce(out["stage_logits"], next_stage)
            l_inf = bce(out["infiltration_logit"], infiltration)
            loss = w_dyn * dyn + w_stage * l_stage + w_inf * l_inf

            if train:
                opt.zero_grad()
                loss.backward()
                if clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
                opt.step()

            k = x.size(0)
            total_loss += loss.item() * k
            parts["dyn"] += dyn.item() * k
            parts["stage"] += l_stage.item() * k
            parts["inf"] += l_inf.item() * k
            n += k

    n = max(n, 1)
    return total_loss / n, {k: v / n for k, v in parts.items()}


@torch.no_grad()
def validation_scores(model, loader, device):
    """Ranking quality of the head the benchmark actually measures."""
    from sklearn.metrics import average_precision_score, roc_auc_score
    model.eval()
    probs, ys = [], []
    for batch in loader:
        out = model(batch["x"].to(device))
        probs.append(torch.sigmoid(out["infiltration_logit"]).cpu().numpy())
        ys.append(batch["infiltration"].numpy())
    p, y = np.concatenate(probs), np.concatenate(ys)
    if len(np.unique(y)) < 2:
        return {"pr_auc": float("nan"), "roc_auc": float("nan")}, p, y
    return ({"pr_auc": float(average_precision_score(y, p)),
             "roc_auc": float(roc_auc_score(y, p))}, p, y)


def fit_temperature(model, loader, device, max_iter=200):
    """Single-scalar temperature fitted on validation, by NLL.

    Scales the logit before the sigmoid. It cannot change the ranking (so AUC
    is untouched) but it fixes over-confidence, which matters when an analyst
    reads the number as a probability.
    """
    model.eval()
    logits, ys = [], []
    with torch.no_grad():
        for batch in loader:
            logits.append(model(batch["x"].to(device))["infiltration_logit"].cpu())
            ys.append(batch["infiltration"])
    if not logits:
        return 1.0
    z = torch.cat(logits)
    y = torch.cat(ys)
    if len(torch.unique(y)) < 2:
        return 1.0

    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=max_iter)
    bce = torch.nn.BCEWithLogitsLoss()

    def closure():
        opt.zero_grad()
        loss = bce(z / torch.exp(log_t), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t).item())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/synthetic_flows.csv")
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--window-seconds", type=int, default=30)
    ap.add_argument("--context-len", type=int, default=CONTEXT_LEN)
    ap.add_argument("--horizon-k", type=int, default=HORIZON_K)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=64)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=SEED)
    # Loss balance. Defaults rebalance toward the infiltration head, which the
    # unweighted sum starved (measured at ~8-12% of total loss).
    ap.add_argument("--w-dyn", type=float, default=0.3)
    ap.add_argument("--w-stage", type=float, default=1.0)
    ap.add_argument("--w-inf", type=float, default=3.0)
    ap.add_argument("--rollout-steps", type=int, default=3,
                    help="R-step autoregressive dynamics loss; 1 = original one-step behaviour")
    ap.add_argument("--clip", type=float, default=1.0, help="gradient-norm clip; 0 disables")
    ap.add_argument("--patience", type=int, default=12, help="early-stopping patience (epochs)")
    ap.add_argument("--lr-patience", type=int, default=4)
    ap.add_argument("--select-on", default="pr_auc", choices=["pr_auc", "roc_auc", "loss"],
                    help="validation metric used to pick the best epoch")
    ap.add_argument("--day-boundaries", default=None,
                    help="Path to a *.day_boundaries.json written by real_data_adapter.py. "
                         "When given, train/val/test is split chronologically WITHIN each day "
                         "and unioned, instead of one 70/15/15 cut across the whole file -- "
                         "needed when different days contain different attack families "
                         "(see dataset.day_aware_split).")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {args.data} ...")
    df = pd.read_csv(args.data)
    states, label_ids, _, _ = build_state_sequence(df, args.window_seconds)
    print(f"Built {len(states)} time windows of {states.shape[1]} features each.")

    if args.day_boundaries:
        with open(resolve_data_path(args.day_boundaries)) as f:
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
    mean = states[train_windows].mean(axis=0)
    std = states[train_windows].std(axis=0) + 1e-6
    states_norm = (states - mean) / std

    full_ds = NetworkStateSequenceDataset(states_norm, label_ids, args.context_len,
                                          args.horizon_k, rollout_steps=args.rollout_steps)

    sets = {"train": set(train_windows), "val": set(val_windows), "test": set(test_windows)}
    idx = {k: [] for k in sets}
    for i, t in enumerate(full_ds.valid_t):           # one pass instead of three
        for k, s in sets.items():
            if t in s:
                idx[k].append(i)
                break
    train_idx, val_idx, test_idx = idx["train"], idx["val"], idx["test"]
    print(f"Sequence examples -> train:{len(train_idx)} val:{len(val_idx)} test:{len(test_idx)}")

    g = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=args.batch_size,
                              shuffle=True, generator=g)
    val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=256, shuffle=False)

    model = WorldModel(N_FEATURES, len(STAGES), hidden_dim=args.hidden_dim,
                       num_layers=args.num_layers).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=args.lr_patience)

    def better(a, b):
        return a < b if args.select_on == "loss" else a > b

    best = float("inf") if args.select_on == "loss" else -float("inf")
    best_state, best_epoch, stale, history = None, 0, 0, []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_parts = run_epoch(model, train_loader, opt, device, args.w_dyn,
                                      args.w_stage, args.w_inf, train=True, clip=args.clip,
                                      rollout_steps=args.rollout_steps)
        val_loss, val_parts = run_epoch(model, val_loader, None, device, args.w_dyn,
                                        args.w_stage, args.w_inf, train=False,
                                        rollout_steps=args.rollout_steps)
        scores, _, _ = validation_scores(model, val_loader, device)
        sched.step(val_loss)

        criterion = val_loss if args.select_on == "loss" else scores[args.select_on]
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": val_loss,
                        "val_pr_auc": scores["pr_auc"], "val_roc_auc": scores["roc_auc"],
                        "lr": opt.param_groups[0]["lr"], **{f"train_{k}": v for k, v in tr_parts.items()}})
        print(f"epoch {epoch:03d}  train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"val_pr_auc={scores['pr_auc']:.4f}  val_roc_auc={scores['roc_auc']:.4f}"
              f"  lr={opt.param_groups[0]['lr']:.2e}")

        if not np.isnan(criterion) and better(criterion, best):
            best, best_epoch, stale = criterion, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                print(f"early stop: no improvement in {args.patience} epochs")
                break

    # Everything written below reflects the BEST epoch, not the last one.
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "world_model.pt"))

    temperature = fit_temperature(model, val_loader, device)
    final, _, _ = validation_scores(model, val_loader, device)
    print(f"\nBest epoch {best_epoch} ({args.select_on}={best:.4f}). "
          f"Calibration temperature {temperature:.3f}.")

    config = {
        "n_features": N_FEATURES,
        "n_stages": len(STAGES),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "context_len": args.context_len,
        "horizon_k": args.horizon_k,
        "window_seconds": args.window_seconds,
        "seed": args.seed,
        "data": args.data,
        "day_boundaries": args.day_boundaries,
        "loss_weights": {"dyn": args.w_dyn, "stage": args.w_stage, "inf": args.w_inf},
        "rollout_steps": args.rollout_steps,
        "best_epoch": best_epoch,
        "select_on": args.select_on,
        "val_pr_auc": final["pr_auc"],
        "val_roc_auc": final["roc_auc"],
        "temperature": temperature,
    }
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    np.savez(os.path.join(args.out_dir, "norm_stats.npz"), mean=mean, std=std)
    with open(os.path.join(args.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    np.savez(os.path.join(args.out_dir, "split_indices.npz"),
             train=np.array(train_idx), val=np.array(val_idx), test=np.array(test_idx))

    print(f"Model + config saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
