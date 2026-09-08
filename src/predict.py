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
"""
import json
import os

import numpy as np
import torch

from features import STATE_FEATURE_NAMES
from mitre_mapping import ID_TO_STAGE, MITRE_TACTIC
from world_model import WorldModel


class InfiltrationPredictor:
    def __init__(self, model_dir="models"):
        with open(os.path.join(model_dir, "config.json")) as f:
            self.cfg = json.load(f)
        norm = np.load(os.path.join(model_dir, "norm_stats.npz"))
        self.mean, self.std = norm["mean"], norm["std"]

        self.device = torch.device("cpu")
        self.model = WorldModel(self.cfg["n_features"], self.cfg["n_stages"],
                                 self.cfg["hidden_dim"], self.cfg["num_layers"]).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(model_dir, "world_model.pt"),
                                                map_location=self.device))
        self.model.eval()

    def normalize(self, raw_states):
        return (raw_states - self.mean) / self.std

    def denormalize(self, norm_states):
        return norm_states * self.std + self.mean

    def _saliency(self, x):
        """Gradient x input over the raw context window for the infiltration logit."""
        x = x.clone().detach().requires_grad_(True)
        out = self.model(x.unsqueeze(0))
        out["infiltration_logit"].backward()
        grad = x.grad.numpy()               # (L, F)
        saliency = np.abs(grad * x.detach().numpy())
        per_feature = saliency.sum(axis=0)  # sum across time steps
        order = np.argsort(-per_feature)
        top = [(STATE_FEATURE_NAMES[i], float(per_feature[i])) for i in order[:8]]
        return top

    def rollout(self, context_states_norm: np.ndarray, k: int = None, explain: bool = True):
        """context_states_norm: (L, F) normalized states, most recent last.
        Returns a dict with per-step forecast (+ explainability artifacts
        unless explain=False, which skips the gradient/saliency pass --
        use that for cheap repeated calls, e.g. scanning a whole timeline).
        """
        k = k or self.cfg["horizon_k"]
        L = self.cfg["context_len"]
        x = torch.tensor(context_states_norm[-L:], dtype=torch.float32)
        if x.shape[0] < L:
            pad = np.repeat(x[0:1].numpy(), L - x.shape[0], axis=0)
            x = torch.tensor(np.concatenate([pad, x.numpy()], axis=0), dtype=torch.float32)

        attention_first = None
        saliency_first = self._saliency(x) if explain else []

        steps = []
        cur = x
        with torch.no_grad():
            for step in range(1, k + 1):
                out = self.model(cur.unsqueeze(0))
                if attention_first is None:
                    attention_first = out["attention"].squeeze(0).numpy().tolist()
                prob = torch.sigmoid(out["infiltration_logit"]).item()
                stage_probs = torch.softmax(out["stage_logits"], dim=-1).squeeze(0).numpy()
                stage_id = int(stage_probs.argmax())
                stage = ID_TO_STAGE[stage_id]
                steps.append({
                    "step": step,
                    "infiltration_probability": prob,
                    "predicted_stage": stage,
                    "mitre_tactic": MITRE_TACTIC[stage],
                    "stage_confidence": float(stage_probs[stage_id]),
                })
                next_state = out["next_state"]                  # (1, F) normalized
                cur = torch.cat([cur[1:], next_state], dim=0)

        return {
            "forecast": steps,
            "attention_weights": attention_first,   # over the L input context steps
            "top_driving_features": saliency_first, # [(name, score), ...]
            "overall_infiltration_risk": max(s["infiltration_probability"] for s in steps),
        }
