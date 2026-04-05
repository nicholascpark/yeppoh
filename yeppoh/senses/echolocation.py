"""Echolocation — active sonar via Genesis Lidar raycasting.

The agent emits a fan of rays from its centroid position.
Returns distance + intensity for each ray.

This is an ACTIVE sense — emitting costs energy (from the metabolism
budget). The agent decides whether to ping via the sensing action.

                              dims: n_rays * 2 (distance + intensity)
                              default 16 rays → 32 dims
"""

from __future__ import annotations

from typing import Any

import torch
import numpy as np

import genesis as gs


class Echolocation:
    """Active sonar via GPU-accelerated raycasting."""

    name = "echolocation"

    def __init__(self, n_rays: int = 16, max_range: float = 5.0):
        self.n_rays = n_rays
        self.max_range = max_range
        self.obs_dim = n_rays * 2  # distance + intensity per ray
        self.sensor = None
        self._active = True  # controlled by agent's sensing action

    def setup(self, scene: Any, entity: Any, cfg: dict) -> None:
        self.n_rays = cfg.get("n_rays", self.n_rays)
        self.max_range = cfg.get("max_range", self.max_range)
        self.obs_dim = self.n_rays * 2

        # Attach a Lidar sensor to the entity
        # Spherical pattern gives omnidirectional sonar
        self.sensor = scene.add_sensor(
            gs.sensors.Lidar(
                entity_idx=entity.idx,
                link_idx_local=0,
                max_range=self.max_range,
                n_horizontal=self.n_rays,
                n_vertical=1,
            ),
        )

    def read(self, env_idx: int | None = None) -> torch.Tensor:
        """Read echolocation returns. (n_envs, n_rays * 2)."""
        if self.sensor is None or not self._active:
            n_envs = 1  # fallback
            return torch.zeros(n_envs, self.obs_dim)

        raw = self.sensor.read()  # (n_envs, n_rays) distances

        # Normalize distances to [0, 1] range
        distances = raw / self.max_range  # (n_envs, n_rays)

        # Intensity: inverse square falloff (closer = stronger return)
        intensity = 1.0 / (distances.clamp(min=0.01) ** 2 + 1.0)

        return torch.cat([distances, intensity], dim=-1)

    def set_active(self, active: bool) -> None:
        """Agent controls whether to emit sonar pulses."""
        self._active = active

    def reset(self) -> None:
        self._active = True
