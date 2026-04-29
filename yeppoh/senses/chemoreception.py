"""Chemoreception — sensing chemical/pheromone gradients.

Reads from the external pheromone diffusion grid (yeppoh.world.pheromones).
For each chemical channel, reports concentration + gradient direction.

                              dims: n_channels * 3 (value + grad_x + grad_y)
                              default 3 channels → 9 dims
"""

from __future__ import annotations

from typing import Any

import torch


class Chemoreception:
    """Chemical gradient sensing from the pheromone grid."""

    name = "chemoreception"

    def __init__(self, n_channels: int = 3):
        self.n_channels = n_channels
        self.obs_dim = n_channels * 3
        self.pheromone_grid = None  # set externally by the environment
        self._position_fn = None  # callable that returns agent position

    def setup(self, scene: Any, entity: Any, cfg: dict) -> None:
        self.n_channels = cfg.get("pheromone_channels", self.n_channels)
        self.obs_dim = self.n_channels * 3
        self.entity = entity

    def set_pheromone_grid(self, grid) -> None:
        """Link to the world's pheromone system. Called by the environment."""
        self.pheromone_grid = grid

    def read(self, env_idx: int | None = None) -> torch.Tensor:
        """Read chemical concentrations and gradients. (n_envs, n_channels * 3)."""
        if self.pheromone_grid is None or self.entity is None:
            n_envs = 1
            return torch.zeros(n_envs, self.obs_dim)

        # Get creature position to sample the grid
        pos = self.entity.get_particles_pos()  # (n_envs, n_particles, 3)
        centroid = pos.mean(dim=1)  # (n_envs, 3)

        # Sample pheromone grid at creature position
        concentration, gradient = self.pheromone_grid.sample(centroid)
        # concentration: (n_envs, n_channels)
        # gradient: (n_envs, n_channels, 3) but we take xy only

        grad_xy = gradient[..., :2]  # (n_envs, n_channels, 2)

        # Flatten: [conc_0, gx_0, gy_0, conc_1, gx_1, gy_1, ...]
        n_envs = centroid.shape[0]
        obs = torch.cat([
            concentration.unsqueeze(-1),  # (n_envs, n_ch, 1)
            grad_xy,                       # (n_envs, n_ch, 2)
        ], dim=-1).reshape(n_envs, -1)

        return obs

    def reset(self) -> None:
        pass
