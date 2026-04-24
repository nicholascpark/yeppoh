"""Metabolic energy system.

Each agent has an energy budget that gates its actions. Energy is:
- Gained from "food" sources in the environment
- Spent on movement, growth, sensing, and signaling
- Depleted over time (basal metabolic cost)

This forces tradeoffs: an agent can't grow, move, AND signal at full
power simultaneously. Specialization emerges from energy pressure.
"""

from __future__ import annotations

import torch


class MetabolicSystem:
    """Per-agent energy bookkeeping."""

    def __init__(
        self,
        n_agents: int,
        n_envs: int = 1,
        max_energy: float = 100.0,
        initial_energy: float = 80.0,
        basal_cost: float = 0.1,
        device: str = "cuda",
    ):
        self.n_agents = n_agents
        self.n_envs = n_envs
        self.max_energy = max_energy
        self.basal_cost = basal_cost
        self.device = device

        self.energy = torch.full(
            (n_envs, n_agents), initial_energy,
            device=device, dtype=torch.float32,
        )

    # ── Cost table ────────────────────────────────────────────────────
    # ★ Tune these to change what behaviors are "expensive"

    COSTS = {
        "move": 0.5,       # per-step locomotion cost
        "grow": 2.0,       # particle emission
        "signal": 0.3,     # pheromone emission
        "echolocate": 0.2, # sonar ping
        "contract": 0.3,   # muscle actuation
    }

    def step(self, actions: dict[str, torch.Tensor]) -> torch.Tensor:
        """Deduct energy costs and apply basal metabolism.

        Args:
            actions: dict of action_type → (n_envs, n_agents) magnitude

        Returns:
            energy_mask: (n_envs, n_agents) float in [0, 1] indicating
                how much of the requested action can be performed given
                available energy. 1.0 = full action, 0.0 = no energy.
        """
        total_cost = torch.full_like(self.energy, self.basal_cost)

        for action_type, magnitude in actions.items():
            cost_rate = self.COSTS.get(action_type, 0.1)
            total_cost += cost_rate * magnitude.abs()

        # How much can we afford?
        affordable = (self.energy / total_cost.clamp(min=1e-8)).clamp(0.0, 1.0)

        # Deduct actual cost (scaled by what we can afford)
        self.energy -= total_cost * affordable
        self.energy.clamp_(0.0, self.max_energy)

        return affordable

    def feed(self, agent_mask: torch.Tensor, amount: float = 10.0) -> None:
        """Add energy to specific agents (found food, absorbing light, etc.)."""
        self.energy += agent_mask.float() * amount
        self.energy.clamp_(0.0, self.max_energy)

    def get_energy(self) -> torch.Tensor:
        """Current energy levels. (n_envs, n_agents)."""
        return self.energy

    def get_energy_fraction(self) -> torch.Tensor:
        """Energy as fraction of max. (n_envs, n_agents) in [0, 1]."""
        return self.energy / self.max_energy

    def is_alive(self) -> torch.Tensor:
        """Which agents still have energy. (n_envs, n_agents) bool."""
        return self.energy > 0.0

    def reset(self, env_idx: torch.Tensor | None = None) -> None:
        if env_idx is None:
            self.energy.fill_(self.max_energy * 0.8)
        else:
            self.energy[env_idx] = self.max_energy * 0.8
