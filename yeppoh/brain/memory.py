"""Temporal memory — GRU-based recurrent state.

Gives agents short-term memory and the ability to generate rhythmic
patterns (oscillations, breathing) from internal dynamics rather
than explicit oscillator parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TemporalMemory(nn.Module):
    """GRU memory module.

    Maintains a hidden state across timesteps, giving the agent
    temporal context and implicit memory.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
        )

    def forward(
        self, x: torch.Tensor, hidden: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Process one timestep.

        Args:
            x: (batch, input_dim)
            hidden: (n_layers, batch, hidden_dim) or None

        Returns:
            output: (batch, hidden_dim)
            hidden: (n_layers, batch, hidden_dim)
        """
        if hidden is None:
            hidden = torch.zeros(
                self.n_layers, x.shape[0], self.hidden_dim,
                device=x.device, dtype=x.dtype,
            )

        # GRU expects (batch, seq_len, input_dim)
        out, hidden = self.gru(x.unsqueeze(1), hidden)
        return out.squeeze(1), hidden

    def initial_state(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(self.n_layers, batch_size, self.hidden_dim, device=device)
