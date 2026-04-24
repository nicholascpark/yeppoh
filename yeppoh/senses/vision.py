"""Vision — egocentric depth camera.

Attaches a small camera to the creature and returns a compressed
feature vector (not raw pixels — that would be too large for the
policy network without a CNN).

We render a small depth image and reduce it to summary statistics:
- Mean depth in each quadrant (4 values)
- Min depth (closest object direction)
- Overall occupancy (how cluttered the view is)
                              dims: 8
"""

from __future__ import annotations

from typing import Any

import torch

import genesis as gs


class Vision:
    """Egocentric depth sensing via Genesis camera."""

    name = "vision"
    obs_dim = 8

    def __init__(self, resolution: int = 32):
        self.resolution = resolution
        self.camera = None

    def setup(self, scene: Any, entity: Any, cfg: dict) -> None:
        self.resolution = cfg.get("vision_resolution", self.resolution)

        self.camera = scene.add_camera(
            gs.cameras.Camera(
                res=(self.resolution, self.resolution),
                pos=(0, 0, 0),
                entity_idx=entity.idx,
                link_idx_local=0,
            ),
        )

    def read(self, env_idx: int | None = None) -> torch.Tensor:
        """Read compressed vision features. (n_envs, 8)."""
        if self.camera is None:
            return torch.zeros(1, self.obs_dim)

        # Read depth buffer
        depth = self.camera.read("depth")  # (n_envs, H, W)
        n_envs = depth.shape[0]
        h, w = depth.shape[1], depth.shape[2]

        # Split into quadrants and compute mean depth
        mid_h, mid_w = h // 2, w // 2
        quadrants = [
            depth[:, :mid_h, :mid_w],      # top-left
            depth[:, :mid_h, mid_w:],       # top-right
            depth[:, mid_h:, :mid_w],       # bottom-left
            depth[:, mid_h:, mid_w:],       # bottom-right
        ]
        quad_means = torch.stack(
            [q.reshape(n_envs, -1).mean(dim=1) for q in quadrants],
            dim=1,
        )  # (n_envs, 4)

        # Global stats
        flat_depth = depth.reshape(n_envs, -1)
        min_depth = flat_depth.min(dim=1, keepdim=True).values
        mean_depth = flat_depth.mean(dim=1, keepdim=True)
        # Occupancy: fraction of pixels closer than half max range
        max_depth = flat_depth.max(dim=1, keepdim=True).values.clamp(min=1e-8)
        occupancy = (flat_depth < max_depth * 0.5).float().mean(dim=1, keepdim=True)
        # Depth variance (texture/complexity of scene)
        depth_var = flat_depth.var(dim=1, keepdim=True)

        return torch.cat([quad_means, min_depth, mean_depth, occupancy, depth_var], dim=-1)

    def reset(self) -> None:
        pass
