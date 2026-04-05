"""CreatureBody — assembled multi-part creature in a Genesis scene.

This is the main interface between the RL environment and Genesis physics.
It wraps the body plan result and provides methods to read state and
apply actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import numpy as np

from .morphology import BodyPartSpec, BodyPlanResult
from .actuators import ActuatorInterface


@dataclass
class AgentCluster:
    """One RL agent's region of the creature."""

    agent_id: str
    part_name: str
    particle_indices: np.ndarray  # which particles this agent controls
    entity: Any = None


class CreatureBody:
    """A creature assembled from Genesis entities.

    Manages the mapping between RL agents and physical body parts.
    Each agent controls a cluster of particles within one body part.
    """

    def __init__(
        self,
        body_plan: BodyPlanResult,
        creature_id: int = 0,
        dt: float = 0.01,
    ):
        self.body_plan = body_plan
        self.creature_id = creature_id
        self.parts = body_plan.parts
        self.actuators = ActuatorInterface(self.parts, dt=dt)

        # Build agent clusters — divide each part's particles among its agents
        self.clusters: list[AgentCluster] = []
        self._build_clusters()

    def _build_clusters(self) -> None:
        """Assign particles to agent clusters via spatial partitioning."""
        agent_idx = 0
        for part in self.parts:
            n_agents = part.n_agents
            # Actual particle assignment happens after scene.build()
            # when we know particle counts. For now, create placeholders.
            for i in range(n_agents):
                self.clusters.append(AgentCluster(
                    agent_id=f"creature_{self.creature_id}_agent_{agent_idx}",
                    part_name=part.name,
                    particle_indices=np.array([]),  # filled in finalize()
                    entity=part.entity,
                ))
                agent_idx += 1

    def finalize(self, n_envs: int) -> None:
        """Called after scene.build() to resolve particle indices.

        Genesis assigns particle indices at build time. This method
        queries each entity for its particle range and divides them
        among the agent clusters.
        """
        for part in self.parts:
            if part.entity is None:
                continue

            # Get particle count for this entity
            try:
                n_particles = part.entity.n_particles
            except AttributeError:
                continue

            part.particle_range = (0, n_particles)

            # Find clusters that belong to this part
            part_clusters = [c for c in self.clusters if c.part_name == part.name]
            if not part_clusters:
                continue

            # Divide particles evenly among clusters
            indices = np.arange(n_particles)
            splits = np.array_split(indices, len(part_clusters))
            for cluster, split in zip(part_clusters, splits):
                cluster.particle_indices = split

    @property
    def agent_ids(self) -> list[str]:
        return [c.agent_id for c in self.clusters]

    @property
    def n_agents(self) -> int:
        return len(self.clusters)

    def get_cluster(self, agent_id: str) -> AgentCluster:
        for c in self.clusters:
            if c.agent_id == agent_id:
                return c
        raise KeyError(f"Unknown agent: {agent_id}")

    def get_positions(self, agent_id: str) -> torch.Tensor:
        """Get particle positions for one agent's cluster.

        Returns shape (n_envs, n_particles, 3).
        """
        cluster = self.get_cluster(agent_id)
        if cluster.entity is None:
            return torch.zeros(1, 1, 3)
        positions = cluster.entity.get_pos()  # (n_envs, total_particles, 3)
        return positions[:, cluster.particle_indices, :]

    def get_velocities(self, agent_id: str) -> torch.Tensor:
        """Get particle velocities for one agent's cluster."""
        cluster = self.get_cluster(agent_id)
        if cluster.entity is None:
            return torch.zeros(1, 1, 3)
        velocities = cluster.entity.get_vel()
        return velocities[:, cluster.particle_indices, :]

    def apply_motor_actions(self, actions: dict[str, torch.Tensor]) -> None:
        """Apply motor actions from all agents.

        Aggregates per-agent actions into per-part actuation commands,
        then sends to Genesis via the actuator interface.
        """
        # Group actions by part
        part_actions: dict[str, list[torch.Tensor]] = {}
        for cluster in self.clusters:
            if cluster.agent_id in actions:
                part_actions.setdefault(cluster.part_name, []).append(
                    actions[cluster.agent_id]
                )

        # Average actions from multiple agents controlling the same part
        merged = {}
        for part_name, action_list in part_actions.items():
            merged[part_name] = torch.stack(action_list).mean(dim=0)

        self.actuators.step(merged)

    def emit_particles(
        self, agent_id: str, rate: float, direction: torch.Tensor
    ) -> None:
        """Trigger particle emission from a growth-tip agent."""
        cluster = self.get_cluster(agent_id)
        part = next(p for p in self.parts if p.name == cluster.part_name)

        if not part.is_emitter or part.entity is None:
            return

        if rate > 0.01:  # threshold to prevent noise
            part.entity.emit(
                speed=rate * 2.0,
            )

    def reset(self) -> None:
        self.actuators.reset()
