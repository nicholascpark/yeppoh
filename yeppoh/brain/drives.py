"""Internal drive system — functional analogs of motivation.

Not emotions — these are internal variables that modulate behavior:
- Hunger: increases when energy is low, drives foraging
- Curiosity: increases with prediction error, drives exploration
- Fear: spikes on sudden stimuli, drives avoidance
- Social: increases with isolation, drives proximity-seeking

Drives are updated each step based on sensory input and energy state.
They're included in the observation vector so the policy can respond.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DriveSystem(nn.Module):
    """Learned internal drive dynamics.

    Takes sensory features + energy level as input, outputs updated
    drive states. The drives persist across steps via a small
    recurrent update.
    """

    N_DRIVES = 4  # hunger, curiosity, fear, social

    def __init__(self, feature_dim: int = 128):
        super().__init__()

        # Drive update network: current drives + features → new drives
        self.update_net = nn.Sequential(
            nn.Linear(self.N_DRIVES + feature_dim, 64),
            nn.ReLU(),
            nn.Linear(64, self.N_DRIVES),
            nn.Sigmoid(),  # drives are in [0, 1]
        )

    def forward(
        self,
        features: torch.Tensor,
        current_drives: torch.Tensor,
        energy_fraction: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update drives based on current state.

        Args:
            features: (batch, feature_dim)
            current_drives: (batch, 4) — [hunger, curiosity, fear, social]
            energy_fraction: (batch, 1) in [0, 1], optional

        Returns:
            new_drives: (batch, 4)
        """
        # Inject energy as a bias on hunger drive
        if energy_fraction is not None:
            hunger_bias = 1.0 - energy_fraction  # low energy → high hunger
            current_drives = current_drives.clone()
            current_drives[:, 0] = current_drives[:, 0] * 0.5 + hunger_bias.squeeze(-1) * 0.5

        x = torch.cat([current_drives, features], dim=-1)
        new_drives = self.update_net(x)

        # Smooth update (exponential moving average to prevent jerky changes)
        alpha = 0.3
        return alpha * new_drives + (1 - alpha) * current_drives

    def initial_drives(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Neutral initial drive state."""
        return torch.full((batch_size, self.N_DRIVES), 0.5, device=device)
