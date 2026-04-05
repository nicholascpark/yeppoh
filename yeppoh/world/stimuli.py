"""Environmental stimuli — light, temperature, obstacles.

These are static or slowly-changing fields that creatures can
sense and respond to (tropisms). They're defined as simple
callable functions over world coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import math


@dataclass
class LightSource:
    """A directional or point light source."""

    position: tuple[float, float, float] = (0.0, 0.0, 5.0)
    intensity: float = 1.0
    is_directional: bool = False
    direction: tuple[float, float, float] = (0.0, 0.0, -1.0)


class StimuliField:
    """Collection of environmental stimuli that creatures can sense."""

    def __init__(self, cfg: dict, device: str = "cuda"):
        self.device = device
        self.lights: list[LightSource] = []
        self.food_sources: list[tuple[float, float, float]] = []

        # Parse config
        for light_cfg in cfg.get("lights", [{"position": [0, 0, 5]}]):
            self.lights.append(LightSource(**light_cfg))

        for food_pos in cfg.get("food_sources", []):
            self.food_sources.append(tuple(food_pos))

    def sample_light(self, positions: torch.Tensor) -> torch.Tensor:
        """Sample light intensity and direction at positions.

        Args:
            positions: (N, 3) world coordinates

        Returns:
            light_obs: (N, 4) — [intensity, dir_x, dir_y, dir_z]
        """
        N = positions.shape[0]
        result = torch.zeros(N, 4, device=self.device)

        for light in self.lights:
            light_pos = torch.tensor(light.position, device=self.device)

            if light.is_directional:
                direction = torch.tensor(light.direction, device=self.device)
                direction = direction / direction.norm().clamp(min=1e-8)
                result[:, 0] += light.intensity
                result[:, 1:] += direction.unsqueeze(0).expand(N, -1)
            else:
                # Point light — intensity falls off with distance
                delta = light_pos.unsqueeze(0) - positions  # (N, 3)
                dist = delta.norm(dim=-1, keepdim=True).clamp(min=0.1)
                intensity = light.intensity / (dist ** 2)
                direction = delta / dist
                result[:, 0:1] += intensity
                result[:, 1:] += direction * intensity

        # Normalize direction
        dir_norm = result[:, 1:].norm(dim=-1, keepdim=True).clamp(min=1e-8)
        result[:, 1:] /= dir_norm

        return result

    def sample_food(self, positions: torch.Tensor, radius: float = 0.5) -> torch.Tensor:
        """Check proximity to food sources.

        Returns:
            food: (N,) — 1.0 if within radius of any food source, 0.0 otherwise
        """
        N = positions.shape[0]
        result = torch.zeros(N, device=self.device)

        for food_pos in self.food_sources:
            fp = torch.tensor(food_pos, device=self.device)
            dist = (positions - fp.unsqueeze(0)).norm(dim=-1)
            result = torch.maximum(result, (dist < radius).float())

        return result

    def step(self) -> None:
        """Update stimuli (e.g., moving light sources, food respawn)."""
        pass  # static for now — override for dynamic environments
