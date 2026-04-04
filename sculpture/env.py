"""Gymnasium environment for RL-driven mesh sculpture.

The agent controls 16 anchor points on an icosphere. At each step it outputs
small 3D displacements for each anchor, which are interpolated via RBF kernels
to smoothly deform all 162 vertices. Laplacian smoothing prevents spikes.

Observation: normalized vertex positions + mesh statistics (492 dims)
Action: anchor displacements scaled to [-0.02, 0.02] (48 dims)
Episode: 200 steps of cumulative deformation
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import trimesh

from .mesh_ops import (
    create_base_mesh,
    farthest_point_sampling,
    compute_rbf_weights,
    apply_deformation,
    laplacian_smooth,
)
from .rewards import compute_reward


class SculptureEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        subdivisions: int = 2,
        n_anchors: int = 16,
        max_steps: int = 200,
        action_scale: float = 0.02,
        rbf_sigma: float = 0.5,
        smooth_iterations: int = 1,
        smooth_factor: float = 0.2,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.n_anchors = n_anchors
        self.max_steps = max_steps
        self.action_scale = action_scale
        self.smooth_iterations = smooth_iterations
        self.smooth_factor = smooth_factor
        self.render_mode = render_mode

        # Base mesh
        self.initial_mesh = create_base_mesh(subdivisions)
        self.n_vertices = len(self.initial_mesh.vertices)

        # Anchor points — evenly distributed via farthest-point sampling
        self.anchor_indices = farthest_point_sampling(
            self.initial_mesh.vertices, n_anchors
        )

        # RBF weights (fixed, based on initial topology)
        self.rbf_weights = compute_rbf_weights(
            self.initial_mesh.vertices,
            self.initial_mesh.vertices[self.anchor_indices],
            sigma=rbf_sigma,
        )

        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(n_anchors * 3,),
            dtype=np.float32,
        )

        obs_dim = self.n_vertices * 3 + 6
        self.observation_space = spaces.Box(
            low=-10.0, high=10.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        # Will be set in reset()
        self.mesh: trimesh.Trimesh | None = None
        self.prev_vertices: np.ndarray | None = None
        self.current_step = 0
        self.history: list[np.ndarray] = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.mesh = self.initial_mesh.copy()
        self.prev_vertices = None
        self.current_step = 0
        self.history = [self.mesh.vertices.copy()]
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1.0, 1.0) * self.action_scale
        anchor_deltas = action.reshape(self.n_anchors, 3)

        self.prev_vertices = self.mesh.vertices.copy()

        # Deform via RBF interpolation
        apply_deformation(self.mesh, anchor_deltas, self.rbf_weights)

        # Smooth to keep surface organic
        if self.smooth_iterations > 0:
            laplacian_smooth(
                self.mesh,
                iterations=self.smooth_iterations,
                factor=self.smooth_factor,
            )

        reward, reward_info = compute_reward(
            self.mesh, self.initial_mesh, self.prev_vertices,
            self.current_step, self.max_steps,
        )

        self.current_step += 1
        self.history.append(self.mesh.vertices.copy())

        terminated = self.current_step >= self.max_steps
        info = {"reward_components": reward_info, "step": self.current_step}

        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self) -> np.ndarray:
        vertices = self.mesh.vertices.copy()

        # Center and normalize
        centroid = vertices.mean(axis=0)
        vertices -= centroid
        scale = np.max(np.abs(vertices)) + 1e-8
        vertices /= scale

        flat_verts = vertices.flatten().astype(np.float32)

        # Compact statistics
        sa_ratio = float(self.mesh.area / self.initial_mesh.area)
        try:
            vol_ratio = float(abs(self.mesh.volume) / abs(self.initial_mesh.volume))
        except Exception:
            vol_ratio = 1.0

        radii = np.linalg.norm(self.mesh.vertices - centroid, axis=1)

        stats = np.array([
            sa_ratio,
            vol_ratio,
            scale,
            radii.std() / (radii.mean() + 1e-8),  # radial CV
            self.current_step / self.max_steps,     # episode progress
            float(self.n_vertices),
        ], dtype=np.float32)

        return np.concatenate([flat_verts, stats])

    def get_mesh_snapshot(self) -> trimesh.Trimesh:
        """Return a copy of the current mesh."""
        return self.mesh.copy()
