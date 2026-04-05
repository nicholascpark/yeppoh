"""CreatureBrain — full policy network composing all cognitive modules.

This is the top-level neural network that maps observations to actions.
It composes: encoder → memory → communication → drives → action heads.

Used by TorchRL/BenchMARL as the policy module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal

from .encoder import SensoryEncoder
from .memory import TemporalMemory
from .communication import CommChannel
from .drives import DriveSystem


class CreatureBrain(nn.Module):
    """Complete agent brain — processes obs, outputs actions.

    Modular design: each component can be enabled/disabled via config.

    Architecture:
        obs → encoder → [memory] → [communication] → [drives] → policy heads
                                                                    ├─ motor (27)
                                                                    ├─ signal (16)
                                                                    └─ sensing (4)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int = 47,
        feature_dim: int = 128,
        hidden_dim: int = 256,
        memory_dim: int = 64,
        message_dim: int = 16,
        use_memory: bool = True,
        use_communication: bool = True,
        use_drives: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.use_memory = use_memory
        self.use_communication = use_communication
        self.use_drives = use_drives

        # Sensory encoder
        self.encoder = SensoryEncoder(obs_dim, feature_dim, hidden_dim)

        # Optional modules
        policy_input_dim = feature_dim

        if use_memory:
            self.memory = TemporalMemory(feature_dim, memory_dim)
            policy_input_dim = memory_dim
        else:
            self.memory = None

        if use_communication:
            self.comm = CommChannel(
                feature_dim=policy_input_dim,
                message_dim=message_dim,
            )
            policy_input_dim += policy_input_dim  # features + comm context
        else:
            self.comm = None

        if use_drives:
            self.drives = DriveSystem(policy_input_dim)
            policy_input_dim += DriveSystem.N_DRIVES
        else:
            self.drives = None

        # Policy head — outputs action mean (std is learned parameter)
        self.policy_head = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh(),  # actions in [-1, 1]
        )

        # Value head (for actor-critic)
        self.value_head = nn.Sequential(
            nn.Linear(policy_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Learnable action std
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(
        self,
        obs: torch.Tensor,
        hidden: torch.Tensor | None = None,
        neighbor_messages: torch.Tensor | None = None,
        neighbor_mask: torch.Tensor | None = None,
        current_drives: torch.Tensor | None = None,
        energy_fraction: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Returns dict with: action_mean, action_std, value, hidden,
                          outgoing_message, drives
        """
        # Encode
        features = self.encoder(obs)

        # Memory
        new_hidden = None
        if self.memory is not None:
            features, new_hidden = self.memory(features, hidden)

        # Communication
        outgoing_msg = None
        if self.comm is not None:
            outgoing_msg = self.comm.encode(features)
            if neighbor_messages is not None:
                comm_context = self.comm.decode(features, neighbor_messages, neighbor_mask)
                features = torch.cat([features, comm_context], dim=-1)
            else:
                features = torch.cat([features, torch.zeros_like(features)], dim=-1)

        # Drives
        new_drives = None
        if self.drives is not None:
            if current_drives is None:
                current_drives = self.drives.initial_drives(obs.shape[0], obs.device)
            new_drives = self.drives(features, current_drives, energy_fraction)
            features = torch.cat([features, new_drives], dim=-1)

        # Action and value
        action_mean = self.policy_head(features)
        value = self.value_head(features)
        action_std = self.log_std.exp().expand_as(action_mean)

        return {
            "action_mean": action_mean,
            "action_std": action_std,
            "value": value,
            "hidden": new_hidden,
            "outgoing_message": outgoing_msg,
            "drives": new_drives,
        }

    def get_action(self, obs: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict]:
        """Sample an action from the policy. For inference."""
        out = self.forward(obs, **kwargs)
        dist = Normal(out["action_mean"], out["action_std"])
        action = dist.sample()
        return action.clamp(-1.0, 1.0), out
