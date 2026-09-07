"""
LSTM-based World Model for network state transition dynamics.

Architecture
------------
  S_{t-L+1..t}  --[per-step Linear encoder]--> [LSTM]--> h_1..h_L
                                                            |
                                          [additive attention over h_1..h_L]
                                                            |
                                                      context vector c
                                                            |
        +-------------------+-----------------------+------+-------------------+
        |                   |                        |                         |
  next_state head    next_stage head       infiltration head (K-step)   (attention weights
  (regression,        (classification,      (binary, sigmoid)            returned for
   Linear -> F)        Linear -> n_stages)                                explainability)

The attention weights over the L context steps, together with a simple
gradient x input saliency over the input features, are what the
explainability module (`explain.py`) surfaces to the analyst: "which past
time steps and which traffic features most drove this prediction".

This is trained as a *dynamics model* (it predicts the next raw state
vector, not just a label) and is additionally supervised with the stage
and infiltration heads so a single forward pass yields everything the
prediction engine needs. K-step-ahead forecasts are produced by
autoregressive rollout in `predict.py`: the model's own predicted next
state is fed back in as input for K steps.
"""
import torch
import torch.nn as nn


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.score = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):  # h: (B, L, H)
        e = self.score(torch.tanh(self.proj(h))).squeeze(-1)   # (B, L)
        alpha = torch.softmax(e, dim=-1)                        # (B, L)
        context = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)   # (B, H)
        return context, alpha


class WorldModel(nn.Module):
    def __init__(self, n_features, n_stages, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.encoder = nn.Linear(n_features, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.attn = AdditiveAttention(hidden_dim)

        self.next_state_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_features)
        )
        self.stage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_stages)
        )
        self.infiltration_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):  # x: (B, L, F)
        z = self.encoder(x)
        h, _ = self.lstm(z)                       # (B, L, H)
        context, alpha = self.attn(h)              # (B, H), (B, L)
        next_state = self.next_state_head(context)
        stage_logits = self.stage_head(context)
        infiltration_logit = self.infiltration_head(context).squeeze(-1)
        return {
            "next_state": next_state,
            "stage_logits": stage_logits,
            "infiltration_logit": infiltration_logit,
            "attention": alpha,          # (B, L) - explainability
        }
