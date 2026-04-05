"""Pheromone diffusion grid — chemical signaling on GPU.

A 3D grid of chemical concentrations that diffuse and decay over time.
Creatures emit pheromones (via actions) and sense gradients (via chemoreception).

This enables stigmergy — indirect communication through the environment,
like ant pheromone trails or slime mold chemical signaling.

The grid runs entirely on torch tensors, independent of Genesis.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class PheromoneGrid:
    """3D diffusion grid for chemical signaling."""

    def __init__(
        self,
        n_channels: int = 3,
        grid_size: int = 64,
        world_bounds: tuple[float, float] = (-3.0, 3.0),
        diffusion_rate: float = 0.1,
        decay_rate: float = 0.01,
        device: str = "cuda",
    ):
        self.n_channels = n_channels
        self.grid_size = grid_size
        self.world_bounds = world_bounds
        self.diffusion_rate = diffusion_rate
        self.decay_rate = decay_rate
        self.device = device

        world_range = world_bounds[1] - world_bounds[0]
        self.cell_size = world_range / grid_size

        # Grid: (n_channels, D, H, W)
        self.grid = torch.zeros(
            n_channels, grid_size, grid_size, grid_size,
            device=device, dtype=torch.float32,
        )

        # 3D Laplacian kernel for diffusion
        kernel = torch.zeros(1, 1, 3, 3, 3, device=device)
        kernel[0, 0, 1, 1, 1] = -6.0
        kernel[0, 0, 0, 1, 1] = 1.0
        kernel[0, 0, 2, 1, 1] = 1.0
        kernel[0, 0, 1, 0, 1] = 1.0
        kernel[0, 0, 1, 2, 1] = 1.0
        kernel[0, 0, 1, 1, 0] = 1.0
        kernel[0, 0, 1, 1, 2] = 1.0
        self.laplacian_kernel = kernel

    def step(self) -> None:
        """Advance diffusion and decay by one timestep."""
        # Diffusion via 3D convolution with Laplacian
        for c in range(self.n_channels):
            channel = self.grid[c].unsqueeze(0).unsqueeze(0)
            laplacian = F.conv3d(channel, self.laplacian_kernel, padding=1)
            self.grid[c] += self.diffusion_rate * laplacian.squeeze()

        # Exponential decay
        self.grid *= (1.0 - self.decay_rate)

        # Clamp to valid range
        self.grid.clamp_(0.0, 10.0)

    def emit(
        self,
        positions: torch.Tensor,
        channel: int,
        amount: float = 1.0,
    ) -> None:
        """Emit pheromone at world positions.

        Args:
            positions: (N, 3) world coordinates
            channel: which chemical channel
            amount: emission intensity
        """
        grid_coords = self._world_to_grid(positions)  # (N, 3) ints

        # Bounds check
        valid = (
            (grid_coords >= 0).all(dim=-1)
            & (grid_coords < self.grid_size).all(dim=-1)
        )
        coords = grid_coords[valid]

        if coords.shape[0] > 0:
            self.grid[channel, coords[:, 0], coords[:, 1], coords[:, 2]] += amount

    def sample(
        self, positions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample concentration and gradient at world positions.

        Args:
            positions: (N, 3) world coordinates

        Returns:
            concentration: (N, n_channels)
            gradient: (N, n_channels, 3) — spatial gradient direction
        """
        grid_coords = self._world_to_grid(positions)
        N = positions.shape[0]

        concentrations = torch.zeros(N, self.n_channels, device=self.device)
        gradients = torch.zeros(N, self.n_channels, 3, device=self.device)

        for i in range(N):
            x, y, z = grid_coords[i].long()
            x = x.clamp(1, self.grid_size - 2)
            y = y.clamp(1, self.grid_size - 2)
            z = z.clamp(1, self.grid_size - 2)

            for c in range(self.n_channels):
                concentrations[i, c] = self.grid[c, x, y, z]
                # Central difference gradient
                gradients[i, c, 0] = (
                    self.grid[c, x + 1, y, z] - self.grid[c, x - 1, y, z]
                ) / (2 * self.cell_size)
                gradients[i, c, 1] = (
                    self.grid[c, x, y + 1, z] - self.grid[c, x, y - 1, z]
                ) / (2 * self.cell_size)
                gradients[i, c, 2] = (
                    self.grid[c, x, y, z + 1] - self.grid[c, x, y, z - 1]
                ) / (2 * self.cell_size)

        return concentrations, gradients

    def _world_to_grid(self, positions: torch.Tensor) -> torch.Tensor:
        """Convert world coordinates to grid indices."""
        normalized = (positions - self.world_bounds[0]) / (
            self.world_bounds[1] - self.world_bounds[0]
        )
        return (normalized * self.grid_size).long()

    def reset(self) -> None:
        self.grid.zero_()
