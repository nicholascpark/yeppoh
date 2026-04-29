"""Touch sensing — contact force detection.

Reads contact forces from Genesis ContactForce sensors attached
to creature body links. Reports force magnitude and direction.

For MPM soft bodies, contact forces aren't exposed per-particle.
We approximate by detecting sudden velocity changes at the surface.

                              dims: 6 (force_xyz + contact_point_xyz)
"""

from __future__ import annotations

from typing import Any

import torch

import genesis as gs


class TouchSense:
    """Surface pressure and contact detection."""

    name = "touch"
    obs_dim = 6

    def __init__(self):
        self.sensor = None
        self.entity = None
        self._prev_vel = None

    def setup(self, scene: Any, entity: Any, cfg: dict) -> None:
        self.entity = entity

        # For rigid bodies, use ContactForce sensor directly
        # For soft bodies, we approximate from velocity changes
        try:
            self.sensor = scene.add_sensor(
                gs.sensors.ContactForce(
                    entity_idx=entity.idx,
                    link_idx_local=0,
                ),
            )
        except Exception:
            # Soft body — fall back to velocity-based approximation
            self.sensor = None

    def read(self, env_idx: int | None = None) -> torch.Tensor:
        """Read touch data. (n_envs, 6)."""
        if self.sensor is not None:
            # Direct contact force reading (rigid bodies)
            force = self.sensor.read()  # (n_envs, 3)
            # Contact point approximation: direction of force
            force_norm = force.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            contact_dir = force / force_norm
            return torch.cat([force, contact_dir], dim=-1)

        # Soft body approximation: detect surface velocity discontinuities
        if self.entity is None:
            return torch.zeros(1, self.obs_dim)

        vel = self.entity.get_particles_vel()  # (n_envs, n_particles, 3)
        n_envs = vel.shape[0]

        if self._prev_vel is None:
            self._prev_vel = vel.clone()
            return torch.zeros(n_envs, self.obs_dim, device=vel.device)

        # Velocity change = proxy for contact impulse
        dv = vel - self._prev_vel  # (n_envs, n_particles, 3)
        impulse_mag = dv.norm(dim=-1)  # (n_envs, n_particles)

        # Find the particle with max impulse (likely contact point)
        max_idx = impulse_mag.argmax(dim=1)  # (n_envs,)
        batch_idx = torch.arange(n_envs, device=vel.device)

        contact_force = dv[batch_idx, max_idx]  # (n_envs, 3)
        contact_point = self.entity.get_particles_pos()[batch_idx, max_idx]  # (n_envs, 3)

        self._prev_vel = vel.clone()

        return torch.cat([contact_force, contact_point], dim=-1)

    def reset(self) -> None:
        self._prev_vel = None
