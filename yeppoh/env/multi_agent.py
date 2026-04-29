"""PettingZoo multi-agent environment wrapping Genesis.

This is the main environment class. It:
1. Wraps a Genesis scene with creatures
2. Exposes a PettingZoo ParallelEnv interface
3. Routes observations and actions per-agent
4. Runs world systems (pheromones, energy) each step
"""

from __future__ import annotations

from typing import Any

import functools
import numpy as np
import torch

from pettingzoo import ParallelEnv
from gymnasium import spaces

from .scene import build_scene
from .agent_manager import AgentManager
from .rewards import CompositeReward


class YeppohEnv(ParallelEnv):
    """Multi-agent creature environment.

    Each agent controls a cell cluster within a creature. Agents observe
    their local sensory state and output motor + communication + sensing
    actions.
    """

    metadata = {"name": "yeppoh_v0", "render_modes": ["human"]}

    def __init__(self, cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.max_steps = cfg.get("scene", {}).get("max_steps", 1000)

        # Build Genesis scene and all subsystems
        components = build_scene(cfg)
        self.scene = components["scene"]
        self.creatures = components["creatures"]
        self.senses = components["senses"]
        self.pheromone_grid = components["pheromone_grid"]
        self.energy_system = components["energy_system"]
        self.stimuli = components["stimuli"]
        self.n_envs = components["n_envs"]
        self.dt = components["dt"]

        # Agent management
        self.agent_mgr = AgentManager(
            self.creatures,
            enable_dynamics=cfg.get("creature", {}).get("dynamic_agents", False),
        )

        # PettingZoo required attributes
        self.possible_agents = list(self.agent_mgr.agent_ids)
        self.agents = list(self.possible_agents)

        # Reward function
        reward_cfg = cfg.get("reward", {"locomotion": 1.0, "survival": 0.5})
        self.reward_fn = CompositeReward(reward_cfg)

        # Spaces (same for all agents in this version)
        self._obs_dim = self._compute_obs_dim()
        self._act_dim = 47  # 27 motor + 16 signal + 4 sensing

        self._step_count = 0
        self._prev_positions = None

    def _compute_obs_dim(self) -> int:
        """Sum observation dims across all active senses + drives + messages."""
        sample_agent = self.possible_agents[0]
        sense_dim = self.senses[sample_agent].total_obs_dim if sample_agent in self.senses else 0
        drive_dim = 4   # hunger, curiosity, fear, social
        message_dim = 32  # 2 neighbors × 16-dim messages
        return sense_dim + drive_dim + message_dim

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=-10.0, high=10.0,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:
        return spaces.Box(
            low=-1.0, high=1.0,
            shape=(self._act_dim,),
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        """Reset environment. Returns {agent_id: obs} dict."""
        self.scene.reset()
        self._step_count = 0
        self._prev_positions = None
        self.agents = list(self.possible_agents)

        # Reset subsystems
        for creature in self.creatures:
            creature.reset()
        for sense_sys in self.senses.values():
            sense_sys.reset()
        if self.pheromone_grid:
            self.pheromone_grid.reset()
        if self.energy_system:
            self.energy_system.reset()
        self.agent_mgr.reset()

        # Collect initial observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_obs(agent_id)

        infos = {agent_id: {} for agent_id in self.agents}
        return observations, infos

    def step(self, actions: dict[str, np.ndarray]):
        """Step all agents simultaneously.

        Args:
            actions: {agent_id: action_array} for all agents

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # Parse actions into motor / signal / sensing components
        motor_actions = {}
        signal_actions = {}
        for agent_id, action in actions.items():
            action_t = torch.tensor(action, dtype=torch.float32)
            motor_actions[agent_id] = action_t[:27]
            signal_actions[agent_id] = action_t[27:43]
            # action_t[43:47] = sensing controls (echolocation active, cone direction)

        # Apply motor actions to Genesis
        for creature in self.creatures:
            creature.apply_motor_actions({
                aid: motor_actions[aid][:5].unsqueeze(0)
                for aid in creature.agent_ids
                if aid in motor_actions
            })

        # Step Genesis physics
        self.scene.step()

        # Step world systems
        if self.pheromone_grid:
            self.pheromone_grid.step()
        if self.energy_system:
            # Compute energy costs from action magnitudes
            energy_actions = {}
            for agent_id, action in actions.items():
                action_t = torch.tensor(action, dtype=torch.float32)
                energy_actions["move"] = action_t[0:5].abs().mean().unsqueeze(0).unsqueeze(0)
            self.energy_system.step(energy_actions)
        if self.stimuli:
            self.stimuli.step()

        self._step_count += 1

        # Collect observations
        observations = {}
        for agent_id in self.agents:
            observations[agent_id] = self._get_obs(agent_id)

        # Compute rewards
        reward_state = self._build_reward_state()
        reward_tensor = self.reward_fn.compute(reward_state)
        rewards = {}
        for i, agent_id in enumerate(self.agents):
            rewards[agent_id] = float(reward_tensor[0, i]) if reward_tensor.dim() > 1 else 0.0

        # Update previous positions
        self._prev_positions = reward_state["positions"].clone()

        # Termination / truncation
        terminated = {agent_id: False for agent_id in self.agents}
        truncated = {
            agent_id: self._step_count >= self.max_steps
            for agent_id in self.agents
        }

        infos = {agent_id: {"step": self._step_count} for agent_id in self.agents}

        # Check agent lifecycle
        self.agent_mgr.check_lifecycle(
            self.energy_system.get_energy() if self.energy_system else None
        )

        return observations, rewards, terminated, truncated, infos

    def _get_obs(self, agent_id: str) -> np.ndarray:
        """Build observation vector for one agent."""
        parts = []

        # Sensory readings
        if agent_id in self.senses:
            reading = self.senses[agent_id].read()
            # reading.flat is (n_envs, total_dim); PettingZoo exposes env 0
            parts.append(reading.flat[0])

        # Internal drives (placeholder — filled by brain module in training)
        drive_obs = torch.zeros(4)
        if self.energy_system:
            agent_idx = self.agents.index(agent_id)
            drive_obs[0] = 1.0 - self.energy_system.get_energy_fraction()[0, agent_idx]
        parts.append(drive_obs)

        # Incoming messages (placeholder — filled by communication in training)
        message_obs = torch.zeros(32)
        parts.append(message_obs)

        obs = torch.cat(parts, dim=-1)

        # Pad or trim to expected size
        if obs.shape[-1] < self._obs_dim:
            obs = torch.cat([obs, torch.zeros(self._obs_dim - obs.shape[-1])], dim=-1)
        elif obs.shape[-1] > self._obs_dim:
            obs = obs[:self._obs_dim]

        obs_np = obs.detach().cpu().numpy().astype(np.float32)
        # Defensive: if physics went unstable for one env, zero non-finite values
        # so the training loop doesn't crash on NaN propagating through the brain.
        return np.nan_to_num(obs_np, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_reward_state(self) -> dict[str, Any]:
        """Collect state needed by reward functions."""
        positions = []
        velocities = []

        for creature in self.creatures:
            for cluster in creature.clusters:
                try:
                    pos = creature.get_positions(cluster.agent_id)
                    vel = creature.get_velocities(cluster.agent_id)
                    positions.append(pos.mean(dim=1))  # centroid
                    velocities.append(vel.mean(dim=1))
                except Exception:
                    positions.append(torch.zeros(self.n_envs, 3))
                    velocities.append(torch.zeros(self.n_envs, 3))

        return {
            "positions": torch.stack(positions, dim=1),  # (n_envs, n_agents, 3)
            "velocities": torch.stack(velocities, dim=1),
            "prev_positions": self._prev_positions,
            "energy": self.energy_system.get_energy() if self.energy_system else None,
        }

    def render(self):
        """Render is handled by Genesis viewer (show_viewer=True in config)."""
        pass

    def close(self):
        pass
