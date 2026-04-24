"""World model — latent dynamics predictor (optional).

Learns to predict the next observation given current state and action.
Prediction error serves as an intrinsic curiosity reward.

Inspired by DreamerV3 but simplified for creature-scale observations.
This is Phase 3+ — the basic system works without it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class WorldModel(nn.Module):
    """Simple latent dynamics model for curiosity-driven exploration.

    Encodes observations into a latent space, predicts next latent
    from current latent + action, and decodes back to observation space.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        latent_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: obs → latent
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Dynamics: latent + action → next latent
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Decoder: latent → obs (for reconstruction loss)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, obs_dim),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def predict_next(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, action], dim=-1)
        return self.dynamics(x)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def curiosity_reward(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """Compute intrinsic curiosity reward from prediction error.

        High prediction error = novel/surprising situation = high reward.
        This drives exploration of unfamiliar states.
        """
        with torch.no_grad():
            latent = self.encode(obs)
            predicted_next = self.predict_next(latent, action)
            actual_next = self.encode(next_obs)

            error = F.mse_loss(predicted_next, actual_next, reduction="none").mean(-1)
            return error.clamp(0.0, 1.0)

    def training_loss(
        self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor
    ) -> torch.Tensor:
        """Combined reconstruction + dynamics loss."""
        latent = self.encode(obs)
        next_latent = self.encode(next_obs)
        predicted_next = self.predict_next(latent, action)

        # Reconstruction loss
        recon = self.decode(latent)
        recon_loss = F.mse_loss(recon, obs)

        # Dynamics prediction loss
        dynamics_loss = F.mse_loss(predicted_next, next_latent.detach())

        return recon_loss + dynamics_loss
