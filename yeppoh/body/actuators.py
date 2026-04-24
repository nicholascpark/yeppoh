"""Muscle actuation interface.

Translates agent motor actions into Genesis actuation signals.
Genesis MPM.Muscle / FEM.Muscle accept a per-step `actu` tensor
that modulates contraction along fiber directions.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import numpy as np


@dataclass
class ActuationCommand:
    """One step of actuation for a body part."""

    contraction: float  # [-1, 1] contraction amplitude
    frequency: float  # modulation frequency (Hz)
    phase: float  # phase offset (radians)
    fiber_dir: tuple[float, float, float] = (0.0, 0.0, 1.0)


class ActuatorInterface:
    """Converts agent motor actions to Genesis actuation tensors.

    The agent outputs high-level commands (contraction amplitude,
    frequency, phase). This class converts them into the per-particle
    actuation signals that Genesis expects.
    """

    def __init__(self, body_parts: list, dt: float = 0.01):
        self.body_parts = body_parts
        self.dt = dt
        self.time = 0.0

    def step(self, motor_actions: dict[str, torch.Tensor]) -> None:
        """Apply motor actions to all body parts.

        Args:
            motor_actions: dict mapping part_name to action tensor.
                Each tensor has shape (n_envs, action_dim) where
                action_dim = 5 (amplitude, freq, phase, wave_dir_x, wave_dir_y)
        """
        self.time += self.dt

        for part in self.body_parts:
            if part.name not in motor_actions:
                continue

            action = motor_actions[part.name]
            actu = self._compute_actuation(action, part)

            if part.entity is not None and hasattr(part.entity, "set_actuation"):
                part.entity.set_actuation(actu)

    def _compute_actuation(
        self, action: torch.Tensor, part
    ) -> torch.Tensor:
        """Convert high-level action to per-particle actuation signal.

        action[:, 0] = contraction amplitude [-1, 1]
        action[:, 1] = frequency [0, 1] → mapped to [0.5, 5.0] Hz
        action[:, 2] = phase offset [0, 1] → mapped to [0, 2pi]
        action[:, 3:5] = peristaltic wave direction (x, y)
        """
        amplitude = action[:, 0]  # (n_envs,)
        freq = action[:, 1] * 4.5 + 0.5  # [0.5, 5.0] Hz
        phase = action[:, 2] * 2.0 * np.pi  # [0, 2pi]

        # Sinusoidal contraction with agent-controlled parameters
        signal = amplitude * torch.sin(2 * np.pi * freq * self.time + phase)

        return signal.unsqueeze(-1)  # (n_envs, 1) — broadcast to all particles

    def reset(self) -> None:
        self.time = 0.0
