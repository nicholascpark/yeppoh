"""Proprioception — the creature's sense of its own body.

Reads body state directly from Genesis particle data:
- Cluster centroid position (3)
- Cluster centroid velocity (3)
- Cluster angular velocity estimate (3)
- Cluster bounding box extents (3)
- Material strain estimate (3)
- Mass / particle count (1)
- Energy level from metabolic system (1)
- Age / timestep (1)
                              total: 18 dims
"""

from __future__ import annotations

from typing import Any

import torch
import numpy as np


class Proprioception:
    """Sense of self — body state from Genesis particle data."""

    name = "proprioception"
    obs_dim = 18

    def __init__(self):
        self.entity = None
        self.particle_indices = None
        self._step_count = 0

    def setup(self, scene: Any, entity: Any, cfg: dict) -> None:
        self.entity = entity

    def read(self, env_idx: int | None = None) -> torch.Tensor:
        """Read proprioceptive state. Returns (n_envs, 18)."""
        self._step_count += 1

        if self.entity is None:
            return torch.zeros(1, self.obs_dim)

        # Particle positions and velocities
        pos = self.entity.get_particles_pos()  # (n_envs, n_particles, 3)
        vel = self.entity.get_particles_vel()  # (n_envs, n_particles, 3)

        n_envs = pos.shape[0]

        # Centroid position
        centroid = pos.mean(dim=1)  # (n_envs, 3)

        # Centroid velocity
        centroid_vel = vel.mean(dim=1)  # (n_envs, 3)

        # Angular velocity estimate (cross product method)
        r = pos - centroid.unsqueeze(1)  # relative positions
        # Approximate angular velocity from particle motions
        cross = torch.cross(r, vel - centroid_vel.unsqueeze(1), dim=-1)
        r_sq = (r * r).sum(dim=-1, keepdim=True).clamp(min=1e-8)
        omega = (cross / r_sq).mean(dim=1)  # (n_envs, 3)

        # Bounding box extents
        bb_min = pos.min(dim=1).values  # (n_envs, 3)
        bb_max = pos.max(dim=1).values
        extents = bb_max - bb_min  # (n_envs, 3)

        # Strain estimate — std of inter-particle distances vs rest
        strain = r.norm(dim=-1).std(dim=1, keepdim=True)  # (n_envs, 1)
        strain = strain.expand(n_envs, 3)  # pad to 3 for consistent dims

        # Scalar features
        mass = torch.full((n_envs, 1), pos.shape[1], dtype=pos.dtype, device=pos.device)
        energy = torch.ones(n_envs, 1, dtype=pos.dtype, device=pos.device)  # placeholder
        age = torch.full(
            (n_envs, 1), self._step_count / 1000.0,
            dtype=pos.dtype, device=pos.device,
        )

        obs = torch.cat([
            centroid, centroid_vel, omega, extents, strain, mass, energy, age
        ], dim=-1)

        return obs

    def reset(self) -> None:
        self._step_count = 0
