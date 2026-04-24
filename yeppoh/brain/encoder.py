"""Sensory encoder — compresses raw observations into features."""

from __future__ import annotations

import torch
import torch.nn as nn


class SensoryEncoder(nn.Module):
    """MLP encoder that compresses raw sensor observations into a feature vector.

    Input: raw obs from all senses (variable size depending on config)
    Output: fixed-size feature vector (feature_dim)
    """

    def __init__(self, obs_dim: int, feature_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )
        self.feature_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode observations. (batch, obs_dim) → (batch, feature_dim)."""
        return self.net(obs)
