"""Experiment runner — config → training loop → results.

Orchestrates:
1. Environment creation (Genesis + PettingZoo)
2. Brain/policy network construction
3. Algorithm setup (TorchRL loss + optimizer)
4. Training loop with logging and checkpointing
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from ..env import YeppohEnv
from ..brain import CreatureBrain
from .algorithms import get_algorithm
from .callbacks import CallbackManager


def run_experiment(cfg: dict | str) -> dict[str, Any]:
    """Run a training experiment from config.

    Args:
        cfg: config dict or path to YAML file

    Returns:
        results dict with training stats
    """
    if isinstance(cfg, str):
        cfg = OmegaConf.to_container(OmegaConf.load(cfg), resolve=True)

    print(f"═══ YEPPOH EXPERIMENT ═══")
    print(f"Creature: {cfg.get('creature', {}).get('plan', 'blob')}")
    print(f"Algorithm: {cfg.get('training', {}).get('algorithm', 'mappo')}")
    print(f"Reward: {list(cfg.get('reward', {}).keys())}")
    print()

    # 1. Create environment
    env = YeppohEnv(cfg)
    n_agents = len(env.possible_agents)
    obs_dim = env._obs_dim
    act_dim = env._act_dim

    print(f"Agents: {n_agents}")
    print(f"Obs dim: {obs_dim}, Act dim: {act_dim}")
    print(f"Parallel envs: {env.n_envs}")

    # 2. Create brain
    brain_cfg = cfg.get("brain", {})
    brain = CreatureBrain(
        obs_dim=obs_dim,
        action_dim=act_dim,
        feature_dim=brain_cfg.get("feature_dim", 128),
        memory_dim=brain_cfg.get("memory_dim", 64),
        message_dim=brain_cfg.get("message_dim", 16),
        use_memory=brain_cfg.get("use_memory", True),
        use_communication=brain_cfg.get("use_communication", True),
        use_drives=brain_cfg.get("use_drives", True),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    brain = brain.to(device)
    print(f"Brain params: {sum(p.numel() for p in brain.parameters()):,}")

    # 3. Algorithm setup
    train_cfg = cfg.get("training", {})
    algo_name = train_cfg.get("algorithm", "mappo")
    algo = get_algorithm(algo_name, train_cfg.get("hyperparams"))

    optimizer = torch.optim.Adam(brain.parameters(), lr=algo.defaults.get("lr", 3e-4))

    # 4. Callbacks
    output_dir = Path(train_cfg.get("output_dir", "runs"))
    output_dir.mkdir(parents=True, exist_ok=True)
    callbacks = CallbackManager(output_dir, cfg)

    # 5. Training loop
    total_timesteps = train_cfg.get("timesteps", 500_000)
    n_steps = algo.defaults.get("n_steps", 2048)
    total_iterations = total_timesteps // (n_steps * env.n_envs)

    print(f"\nTraining for {total_timesteps:,} timesteps ({total_iterations} iterations)")
    print(f"Output: {output_dir}")
    print(f"═══════════════════════\n")

    results = {"rewards": [], "iterations": 0}

    for iteration in range(total_iterations):
        # Collect rollout
        rollout_rewards = _collect_rollout(env, brain, n_steps, device)
        results["rewards"].extend(rollout_rewards)
        results["iterations"] = iteration + 1

        # TODO: compute advantages, run PPO/MADDPG/QMIX update
        # This is where TorchRL/BenchMARL loss modules plug in.
        # For now, this is the integration point.

        callbacks.on_iteration(iteration, results)

        if (iteration + 1) % 50 == 0:
            # Checkpoint
            ckpt_path = output_dir / f"checkpoint_{iteration + 1}.pt"
            torch.save({
                "brain": brain.state_dict(),
                "optimizer": optimizer.state_dict(),
                "iteration": iteration + 1,
                "config": cfg,
            }, ckpt_path)

    # Final save
    torch.save({
        "brain": brain.state_dict(),
        "config": cfg,
        "results": results,
    }, output_dir / "final.pt")

    env.close()
    callbacks.close()
    return results


def _collect_rollout(
    env: YeppohEnv,
    brain: CreatureBrain,
    n_steps: int,
    device: torch.device,
) -> list[float]:
    """Collect n_steps of experience from the environment."""
    episode_rewards = []
    step_reward_sum = 0.0

    obs_dict, _ = env.reset()

    for step in range(n_steps):
        actions = {}
        for agent_id, obs in obs_dict.items():
            obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            action, _ = brain.get_action(obs_tensor)
            actions[agent_id] = action.squeeze(0).cpu().numpy()

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        step_reward_sum += sum(rewards.values())

        # Check for episode end
        if any(truncated.values()) or any(terminated.values()):
            episode_rewards.append(step_reward_sum)
            step_reward_sum = 0.0
            obs_dict, _ = env.reset()

    return episode_rewards
