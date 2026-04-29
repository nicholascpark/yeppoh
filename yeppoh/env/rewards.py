"""Reward functions for creature behavior.

★ THIS IS A KEY EXPERIMENT SURFACE.

Each reward function takes the current state and returns per-agent rewards.
Register new rewards and combine them via config.

Rewards are split into:
- Creature-level (shared by all agents in a creature): locomotion, growth
- Agent-level (individual): energy efficiency, survival
- Cross-creature (competition/cooperation): territory, resource access
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch
import numpy as np


class RewardFunction(ABC):
    """Base class for reward functions."""

    name: str = "unnamed"
    level: str = "creature"  # "creature", "agent", or "global"

    @abstractmethod
    def compute(self, state: dict[str, Any]) -> torch.Tensor:
        """Compute reward.

        Args:
            state: dict with keys:
                - positions: (n_envs, n_agents, 3) agent centroids
                - velocities: (n_envs, n_agents, 3) agent centroid velocities
                - energy: (n_envs, n_agents) energy levels
                - surface_area: (n_envs,) per-creature surface area
                - prev_positions: previous step positions
                - creature_map: agent_idx → creature_idx

        Returns:
            rewards: (n_envs, n_agents)
        """


class LocomotionReward(RewardFunction):
    """Reward creature for moving forward."""

    name = "locomotion"
    level = "creature"

    def __init__(self, direction: tuple[float, float, float] = (1.0, 0.0, 0.0)):
        self.direction = None
        self._direction_tuple = direction

    def compute(self, state: dict[str, Any]) -> torch.Tensor:
        pos = state["positions"]  # (n_envs, n_agents, 3)
        prev_pos = state.get("prev_positions")

        if prev_pos is None:
            return torch.zeros(pos.shape[:2], device=pos.device)

        if self.direction is None:
            self.direction = torch.tensor(
                self._direction_tuple, device=pos.device, dtype=pos.dtype,
            )

        # Per-creature centroid movement in target direction
        displacement = pos.mean(dim=1) - prev_pos.mean(dim=1)  # (n_envs, 3)
        forward = (displacement * self.direction).sum(dim=-1)  # (n_envs,)

        # Broadcast to all agents in the creature
        return forward.unsqueeze(1).expand_as(pos[:, :, 0])


class GrowthReward(RewardFunction):
    """Reward interesting morphological changes."""

    name = "growth"
    level = "creature"

    def compute(self, state: dict[str, Any]) -> torch.Tensor:
        pos = state["positions"]
        n_envs, n_agents, _ = pos.shape

        # Radial variance — reward asymmetric, interesting shapes
        centroid = pos.mean(dim=1, keepdim=True)
        radii = (pos - centroid).norm(dim=-1)  # (n_envs, n_agents)
        radial_cv = radii.std(dim=1) / radii.mean(dim=1).clamp(min=1e-8)

        # Spread — reward expansion
        spread = radii.mean(dim=1)

        reward = radial_cv * 0.5 + spread * 0.3
        return reward.unsqueeze(1).expand(n_envs, n_agents)


class SurvivalReward(RewardFunction):
    """Reward individual agents for staying alive and efficient."""

    name = "survival"
    level = "agent"

    def compute(self, state: dict[str, Any]) -> torch.Tensor:
        energy = state.get("energy")
        pos = state["positions"]  # (n_envs, n_agents, 3)
        if energy is None:
            return torch.zeros(pos.shape[:2], device=pos.device, dtype=pos.dtype)

        # Reward for maintaining energy above critical threshold
        alive_bonus = (energy > 10.0).float() * 0.1
        # Small penalty for energy waste (encourage efficiency)
        efficiency = -0.01 * (energy > 90.0).float()

        return alive_bonus + efficiency


class CoordinationReward(RewardFunction):
    """Reward agents for coordinated movement within a creature."""

    name = "coordination"
    level = "creature"

    def compute(self, state: dict[str, Any]) -> torch.Tensor:
        vel = state.get("velocities")
        pos = state["positions"]  # (n_envs, n_agents, 3)
        if vel is None:
            return torch.zeros(pos.shape[:2], device=pos.device, dtype=pos.dtype)

        # Reward when all agents in a creature move in similar directions
        # (coherent movement vs. incoherent jitter)
        mean_vel = vel.mean(dim=1, keepdim=True)  # (n_envs, 1, 3)
        vel_norm = vel.norm(dim=-1).clamp(min=1e-8)
        mean_norm = mean_vel.norm(dim=-1).clamp(min=1e-8)

        # Cosine similarity between each agent's velocity and group mean
        cos_sim = (vel * mean_vel).sum(dim=-1) / (vel_norm * mean_norm.squeeze(1))

        return cos_sim * 0.3  # (n_envs, n_agents)


# ── Registry ──────────────────────────────────────────────────────────
# ★ Add new reward functions here.

REWARD_REGISTRY: dict[str, type[RewardFunction]] = {
    "locomotion": LocomotionReward,
    "growth": GrowthReward,
    "survival": SurvivalReward,
    "coordination": CoordinationReward,
}


class CompositeReward:
    """Combines multiple reward functions with configurable weights."""

    def __init__(self, reward_cfg: dict):
        self.rewards: list[tuple[RewardFunction, float]] = []
        for name, weight in reward_cfg.items():
            if name in REWARD_REGISTRY:
                self.rewards.append((REWARD_REGISTRY[name](), weight))

    def compute(self, state: dict[str, Any]) -> torch.Tensor:
        total = None
        for reward_fn, weight in self.rewards:
            r = reward_fn.compute(state) * weight
            total = r if total is None else total + r
        return total if total is not None else torch.zeros(1, 1)
