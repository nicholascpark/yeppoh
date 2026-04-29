"""Single-agent Gym-style creature environment.

One MPM soft-body, one policy. Fully vectorized across `n_envs`
Genesis simulations — unlike the MARL path, every env contributes
to the gradient signal.

    reset() -> obs: (n_envs, obs_dim)
    step(action) -> (obs, reward, done, info)

    action: (n_envs, act_dim) in [-1, 1]
    reward: (n_envs,) float32
    done:   (n_envs,) bool
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import genesis as gs

from ..body import build_body
from ..body.actuators import ActuatorInterface
from ..senses.proprioception import Proprioception


class SoloYeppohEnv:
    """Gym-style single-agent soft-body creature env."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        scene_cfg = cfg.get("scene", {})
        creature_cfg = cfg.get("creature", {})

        self.n_envs = scene_cfg.get("n_envs", 16)
        self.dt = scene_cfg.get("dt", 0.01)
        self.max_steps = scene_cfg.get("max_steps", 200)
        self._reward_name = cfg.get("reward", "locomotion")

        # Genesis init — safe to call repeatedly in same process
        backend = gs.gpu if scene_cfg.get("gpu", True) else gs.cpu
        try:
            gs.init(backend=backend)
        except Exception:
            pass  # already initialized

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(
                dt=self.dt,
                substeps=scene_cfg.get("substeps", 32),
            ),
            show_viewer=scene_cfg.get("show_viewer", False),
        )
        self.scene.add_entity(gs.morphs.Plane())

        # Body: force n_agents=1 — solo is a single-policy contract
        plan_name = creature_cfg.get("plan", "blob")
        plan_cfg = {**creature_cfg.get("params", {"radius": 0.3}), "n_agents": 1}
        self.body_plan = build_body(self.scene, plan_name, plan_cfg)
        self.entity = self.body_plan.parts[0].entity

        self.scene.build(n_envs=self.n_envs)

        self.actuator = ActuatorInterface(self.body_plan.parts, dt=self.dt)

        self.prop = Proprioception()
        self.prop.setup(self.scene, self.entity, {})

        self.obs_dim = self.prop.obs_dim  # 18
        self.act_dim = 5  # matches ActuatorInterface: amp, freq, phase, wave_x, wave_y

        self._step_count = 0
        self._prev_centroid: torch.Tensor | None = None
        self._initial_centroid: torch.Tensor | None = None

    # ── Gym-style API ────────────────────────────────────────────────

    def reset(self) -> np.ndarray:
        self.scene.reset()
        self.prop.reset()
        self.actuator.reset()
        self._step_count = 0
        self._prev_centroid = None
        self._initial_centroid = None
        return self._get_obs()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        action_t = torch.tensor(action, dtype=torch.float32)
        self.actuator.step({self.body_plan.parts[0].name: action_t})

        self.scene.step()
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        done = np.full(self.n_envs, self._step_count >= self.max_steps, dtype=bool)
        info = {"step": self._step_count}
        return obs, reward, done, info

    def close(self) -> None:
        pass

    # ── Internals ────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        reading = self.prop.read()  # (n_envs, 18)
        obs_np = reading.detach().cpu().numpy().astype(np.float32)

        # Proprioception slot 15 is raw particle count (~112K for the blob).
        # Unnormalized extremes wreck first-layer activations — compress to ~[0, 1].
        obs_np[:, 15] = np.log1p(obs_np[:, 15]) / 15.0

        return np.nan_to_num(obs_np, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_reward(self) -> np.ndarray:
        pos = self.entity.get_particles_pos()  # (n_envs, n_particles, 3)
        centroid = pos.mean(dim=1)  # (n_envs, 3)

        if self._prev_centroid is None:
            self._prev_centroid = centroid.clone()
            self._initial_centroid = centroid.clone()
            return np.zeros(self.n_envs, dtype=np.float32)

        reward = self._reward_from_name(centroid)

        self._prev_centroid = centroid.clone()
        return np.nan_to_num(reward, nan=0.0, posinf=0.0, neginf=0.0)

    def _reward_from_name(self, centroid: torch.Tensor) -> np.ndarray:
        """★ Extend here to add new reward families (rhythmic, stimulus, aesthetic...)."""
        if self._reward_name == "locomotion":
            # +X displacement per step (dense, easy gradient)
            dx = centroid[:, 0] - self._prev_centroid[:, 0]
            return dx.detach().cpu().numpy().astype(np.float32)

        if self._reward_name == "none":
            return np.zeros(self.n_envs, dtype=np.float32)

        raise ValueError(f"Unknown reward: {self._reward_name}")
