"""Small actor-critic policy for solo training.

Tiny MLP. Gaussian action head with learned log-std. Tanh-squashed mean
keeps actions centered in [-1, 1] without the log-prob bookkeeping of
full tanh-squashing. Final action is sampled from the Gaussian then
clipped to the valid range at apply-time.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class SoloPolicy(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()

        def mlp(out_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(obs_dim, hidden),
                nn.LayerNorm(hidden),
                nn.Tanh(),
                nn.Linear(hidden, hidden),
                nn.Tanh(),
                nn.Linear(hidden, out_dim),
            )

        self.actor_mean = mlp(act_dim)
        self.critic = mlp(1)
        # Start with std ~ 0.6 — enough exploration, not wild
        self.log_std = nn.Parameter(torch.zeros(act_dim) - 0.5)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = torch.tanh(self.actor_mean(obs))
        std = self.log_std.exp().expand_as(mean)
        value = self.critic(obs).squeeze(-1)
        return mean, std, value

    def act(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action for rollout collection."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        return action.clamp(-1.0, 1.0), log_prob, value

    def evaluate(
        self, obs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Log-prob, value, entropy at given (obs, action). For PPO update."""
        mean, std, value = self.forward(obs)
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().sum(-1)
        return log_prob, value, entropy
