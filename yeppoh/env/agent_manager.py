"""Agent lifecycle manager — handles cell cluster split/merge.

Manages the dynamic set of agents within creatures. Agents can:
- Split: when a cluster grows beyond a threshold, spawn a child agent
- Merge: when two adjacent clusters are too small, fuse them
- Die: when energy reaches zero

For Phase 1, agent count is fixed. Split/merge is Phase 3+.
The interface is here so the rest of the system is ready.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np

from ..body.creature import CreatureBody, AgentCluster


@dataclass
class AgentEvent:
    """Records an agent lifecycle event."""

    event_type: str  # "split", "merge", "die", "spawn"
    agent_id: str
    child_id: str | None = None
    env_idx: int | None = None


class AgentManager:
    """Manages the dynamic set of agents across all creatures."""

    def __init__(self, creatures: list[CreatureBody], enable_dynamics: bool = False):
        self.creatures = creatures
        self.enable_dynamics = enable_dynamics
        self.events: list[AgentEvent] = []

        # Collect all agent IDs
        self._agent_ids = []
        for creature in creatures:
            self._agent_ids.extend(creature.agent_ids)

    @property
    def agent_ids(self) -> list[str]:
        return list(self._agent_ids)

    @property
    def n_agents(self) -> int:
        return len(self._agent_ids)

    def get_creature_for_agent(self, agent_id: str) -> CreatureBody:
        """Find which creature an agent belongs to."""
        for creature in self.creatures:
            if agent_id in creature.agent_ids:
                return creature
        raise KeyError(f"Agent {agent_id} not found in any creature")

    def get_teammates(self, agent_id: str) -> list[str]:
        """Get other agents in the same creature."""
        creature = self.get_creature_for_agent(agent_id)
        return [aid for aid in creature.agent_ids if aid != agent_id]

    def check_lifecycle(self, energy: torch.Tensor) -> list[AgentEvent]:
        """Check for split/merge/death conditions.

        Called each step. Returns list of events that occurred.
        Currently a stub — enable_dynamics=True activates in later phases.
        """
        if not self.enable_dynamics:
            return []

        events = []
        # Future: check particle counts for split threshold,
        # check energy for death, check proximity for merge
        self.events.extend(events)
        return events

    def reset(self) -> None:
        self.events.clear()
