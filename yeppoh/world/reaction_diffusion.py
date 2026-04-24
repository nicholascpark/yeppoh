"""Reaction-diffusion patterns — Gray-Scott model on GPU.

Generates Turing patterns (spots, stripes, labyrinths) on the creature
surface. These can drive visual appearance and material property
variation, making creatures look organic without explicit design.

The patterns are computed on a 2D grid and mapped to particle surfaces.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


class ReactionDiffusion:
    """Gray-Scott reaction-diffusion system.

    Two chemicals U and V interact:
        dU/dt = Du * laplacian(U) - U*V^2 + f*(1-U)
        dV/dt = Dv * laplacian(V) + U*V^2 - (f+k)*V

    Parameters f (feed rate) and k (kill rate) control the pattern type:
        f=0.055, k=0.062 → spots
        f=0.039, k=0.058 → stripes
        f=0.026, k=0.051 → waves
        f=0.078, k=0.061 → worms

    ★ Experiment with f and k values for different aesthetics.
    """

    def __init__(
        self,
        grid_size: int = 128,
        Du: float = 0.16,
        Dv: float = 0.08,
        f: float = 0.055,
        k: float = 0.062,
        dt: float = 1.0,
        device: str = "cuda",
    ):
        self.grid_size = grid_size
        self.Du = Du
        self.Dv = Dv
        self.f = f
        self.k = k
        self.dt = dt
        self.device = device

        # Initialize U=1 everywhere, V=0 with random seed patches
        self.U = torch.ones(1, 1, grid_size, grid_size, device=device)
        self.V = torch.zeros(1, 1, grid_size, grid_size, device=device)

        # Seed: small random patches of V
        self._seed_random_patches(n_patches=5, patch_size=8)

        # 2D Laplacian kernel
        kernel = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=torch.float32, device=device,
        ).reshape(1, 1, 3, 3)
        self.laplacian_kernel = kernel

    def _seed_random_patches(self, n_patches: int = 5, patch_size: int = 8) -> None:
        for _ in range(n_patches):
            x = torch.randint(0, self.grid_size - patch_size, (1,)).item()
            y = torch.randint(0, self.grid_size - patch_size, (1,)).item()
            self.V[0, 0, x:x + patch_size, y:y + patch_size] = 1.0

    def step(self, n_steps: int = 10) -> None:
        """Advance the reaction-diffusion system.

        Run multiple sub-steps per environment step for pattern development.
        """
        for _ in range(n_steps):
            lap_U = F.conv2d(self.U, self.laplacian_kernel, padding=1)
            lap_V = F.conv2d(self.V, self.laplacian_kernel, padding=1)

            UVV = self.U * self.V * self.V

            self.U += self.dt * (self.Du * lap_U - UVV + self.f * (1.0 - self.U))
            self.V += self.dt * (self.Dv * lap_V + UVV - (self.f + self.k) * self.V)

            self.U.clamp_(0.0, 1.0)
            self.V.clamp_(0.0, 1.0)

    def sample(self, uv_coords: torch.Tensor) -> torch.Tensor:
        """Sample pattern values at UV coordinates.

        Args:
            uv_coords: (N, 2) in [0, 1] range

        Returns:
            pattern: (N, 2) — U and V concentrations
        """
        # Bilinear sample from grid
        grid = uv_coords * 2.0 - 1.0  # normalize to [-1, 1] for grid_sample
        grid = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, N, 2)

        u_val = F.grid_sample(self.U, grid, align_corners=True, mode="bilinear")
        v_val = F.grid_sample(self.V, grid, align_corners=True, mode="bilinear")

        return torch.cat([
            u_val.squeeze(), v_val.squeeze()
        ], dim=-1)  # (N, 2)

    def get_pattern_image(self) -> torch.Tensor:
        """Return the current V channel as a 2D image for visualization."""
        return self.V.squeeze()  # (grid_size, grid_size)

    def reset(self) -> None:
        self.U.fill_(1.0)
        self.V.fill_(0.0)
        self._seed_random_patches()
