"""
Infiltration Prediction Engine.

Given the last L observed network states, performs K-step autoregressive
forward simulation with the trained world model: at each step the model's
own predicted next state is fed back in as the newest context step
("rollout"), producing:

  - a per-step infiltration probability (the time-series forecast),
  - the predicted MITRE ATT&CK stage at each rolled-out step,
  - attention weights over the input context (explainability),
  - a gradient x input saliency ranking of which state features drove
    the *first* forecast step's infiltration score (explainability).

Batched entry points (`score_batch`, `rollout_batch`) evaluate many context
anchors in one forward pass. Scanning a whole capture for the risk timeline
is otherwise thousands of batch-of-one calls, which is dominated by Python
and torch dispatch overhead rather than by arithmetic.
"""
import json
import os
from typing import List, Optional

import numpy as np
import torch

from .features import STATE_FEATURE_NAMES
from .mitre_mapping import ID_TO_STAGE, MITRE_TACTIC
from .world_model import WorldModel

TOP_FEATURES = 8  # how many saliency-ranked features to surface


class InfiltrationPredictor:
    def __init__(self, model_dir: str = "models", device: Optional[str] = None):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        norm = np.load(os.path.join(model_dir, "norm_stats.npz"))
        self.mean, self.std = norm["mean"], norm["std"]

        self.model_dir = model_dir
        # Temperature scaling fitted on validation at train time. Divides the
        # logit before the sigmoid, so it corrects over-confidence without
        # touching the ranking (AUC is unchanged). Checkpoints trained before
        # calibration existed have no entry and default to 1.0 -- a no-op.
        self.temperature = float(self.cfg.get("temperature", 1.0)) or 1.0
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = WorldModel(self.cfg["n_features"], self.cfg["n_stages"],
                                self.cfg["hidden_dim"], self.cfg["num_layers"]).to(self.device)
        state = torch.load(os.path.join(model_dir, "world_model.pt"),
                           map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.eval()

    # ------------------------------------------------------------------ utils

    def _prob(self, logit: torch.Tensor) -> torch.Tensor:
        """Calibrated probability from a raw infiltration logit."""
        return torch.sigmoid(logit / self.temperature)

    def normalize(self, raw_states):
        return (raw_states - self.mean) / self.std

    def denormalize(self, norm_states):
        return norm_states * self.std + self.mean

    def _as_context(self, context_states_norm: np.ndarray) -> torch.Tensor:
        """(<=L, F) -> (L, F) tensor, left-padded by repeating the oldest state."""
        L = self.cfg["context_len"]
        x = torch.as_tensor(np.asarray(context_states_norm, dtype=np.float32)[-L:],
                            dtype=torch.float32, device=self.device)
        if x.shape[0] < L:
            pad = x[0:1].expand(L - x.shape[0], -1)
            x = torch.cat([pad, x], dim=0)
        return x

    # -------------------------------------------------------------- batch API

    def score_batch(self, contexts: np.ndarray, batch_size: int = 1024) -> dict:
        """Score many (L, F) context windows in one shot.

        `contexts`: (B, L, F) normalized states. Returns per-anchor
        infiltration probability, predicted stage id and stage confidence --
        a single forward pass per chunk rather than B of them.
        """
        x_all = torch.as_tensor(np.asarray(contexts, dtype=np.float32), device=self.device)
        if x_all.ndim == 2:
            x_all = x_all.unsqueeze(0)

        probs, stage_ids, stage_conf = [], [], []
        with torch.no_grad():
            for i in range(0, len(x_all), batch_size):
                out = self.model(x_all[i:i + batch_size])
                probs.append(self._prob(out["infiltration_logit"]).cpu().numpy())
                sp = torch.softmax(out["stage_logits"], dim=-1)
                conf, ids = sp.max(dim=-1)
                stage_ids.append(ids.cpu().numpy())
                stage_conf.append(conf.cpu().numpy())

        return {
            "infiltration_probability": np.concatenate(probs),
            "stage_id": np.concatenate(stage_ids),
            "stage_confidence": np.concatenate(stage_conf),
        }

    def rollout_batch(self, contexts: np.ndarray, k: Optional[int] = None,
                      batch_size: int = 1024) -> dict:
        """K-step autoregressive rollout for a batch of context windows.

        Returns arrays shaped (B, K). Rollout is sequential in K by
        construction, but every anchor advances together, so the whole
        horizon costs K forward passes instead of B*K.
        """
        k = k or self.cfg["horizon_k"]
        x_all = torch.as_tensor(np.asarray(contexts, dtype=np.float32), device=self.device)
        if x_all.ndim == 2:
            x_all = x_all.unsqueeze(0)

        probs_out, stage_out, conf_out = [], [], []
        with torch.no_grad():
            for i in range(0, len(x_all), batch_size):
                cur = x_all[i:i + batch_size]
                p_steps, s_steps, c_steps = [], [], []
                for _ in range(k):
                    out = self.model(cur)
                    p_steps.append(self._prob(out["infiltration_logit"]))
                    sp = torch.softmax(out["stage_logits"], dim=-1)
                    conf, ids = sp.max(dim=-1)
                    s_steps.append(ids)
                    c_steps.append(conf)
                    cur = torch.cat([cur[:, 1:], out["next_state"].unsqueeze(1)], dim=1)
                probs_out.append(torch.stack(p_steps, dim=1).cpu().numpy())
                stage_out.append(torch.stack(s_steps, dim=1).cpu().numpy())
                conf_out.append(torch.stack(c_steps, dim=1).cpu().numpy())

        return {
            "infiltration_probability": np.concatenate(probs_out),   # (B, K)
            "stage_id": np.concatenate(stage_out),                   # (B, K)
            "stage_confidence": np.concatenate(conf_out),            # (B, K)
        }

    def make_contexts(self, states_norm: np.ndarray, anchors) -> np.ndarray:
        """Stack (L, F) context windows ending at each anchor index."""
        L = self.cfg["context_len"]
        return np.stack([states_norm[max(0, t - L + 1): t + 1] if t - L + 1 >= 0
                         else np.concatenate([np.repeat(states_norm[0:1], L - (t + 1), axis=0),
                                              states_norm[: t + 1]], axis=0)
                         for t in anchors]).astype(np.float32)

    # ------------------------------------------------------- explainability

    def _saliency(self, x: torch.Tensor) -> List[tuple]:
        """Gradient x input over the raw context window for the infiltration logit."""
        x = x.clone().detach().requires_grad_(True)
        out = self.model(x.unsqueeze(0))
        out["infiltration_logit"].backward()
        grad = x.grad.detach().cpu().numpy()                 # (L, F)
        saliency = np.abs(grad * x.detach().cpu().numpy())
        per_feature = saliency.sum(axis=0)                   # sum across time steps
        order = np.argsort(-per_feature)
        return [(STATE_FEATURE_NAMES[i], float(per_feature[i])) for i in order[:TOP_FEATURES]]

    # ------------------------------------------------------------ single API

    def rollout(self, context_states_norm: np.ndarray, k: Optional[int] = None,
                explain: bool = True) -> dict:
        """context_states_norm: (L, F) normalized states, most recent last.

        Returns a dict with the per-step forecast (+ explainability artifacts
        unless explain=False, which skips the gradient/saliency pass -- use
        that for cheap repeated calls, though `score_batch` is better still
        when scanning a whole timeline).
        """
        k = k or self.cfg["horizon_k"]
        x = self._as_context(context_states_norm)

        saliency_first = self._saliency(x) if explain else []

        attention_first = None
        steps = []
        cur = x
        with torch.no_grad():
            for step in range(1, k + 1):
                out = self.model(cur.unsqueeze(0))
                if attention_first is None:
                    attention_first = out["attention"].squeeze(0).cpu().numpy().tolist()
                prob = self._prob(out["infiltration_logit"]).item()
                stage_probs = torch.softmax(out["stage_logits"], dim=-1).squeeze(0).cpu().numpy()
                stage_id = int(stage_probs.argmax())
                stage = ID_TO_STAGE[stage_id]
                steps.append({
                    "step": step,
                    "infiltration_probability": prob,
                    "predicted_stage": stage,
                    "mitre_tactic": MITRE_TACTIC[stage],
                    "stage_confidence": float(stage_probs[stage_id]),
                })
                cur = torch.cat([cur[1:], out["next_state"]], dim=0)

        return {
            "forecast": steps,
            "attention_weights": attention_first,   # over the L input context steps
            "top_driving_features": saliency_first, # [(name, score), ...]
            # NOTE: max, not mean -- the peak risk anywhere in the horizon.
            "overall_infiltration_risk": max(s["infiltration_probability"] for s in steps),
        }
