"""Video recording — capture training episodes as MP4."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import numpy as np


def record_episode(
    env: Any,
    brain: Any,
    output_path: str | Path,
    max_steps: int = 500,
    device: str = "cuda",
) -> Path:
    """Record one episode as video using Genesis recorder.

    Args:
        env: YeppohEnv instance
        brain: CreatureBrain model
        output_path: where to save MP4
        max_steps: max steps to record
        device: torch device
    """
    import genesis as gs

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Start Genesis recording
    env.scene.start_recording()

    obs_dict, _ = env.reset()
    total_reward = 0.0

    for step in range(max_steps):
        actions = {}
        for agent_id, obs in obs_dict.items():
            obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                action, _ = brain.get_action(obs_t)
            actions[agent_id] = action.squeeze(0).cpu().numpy()

        obs_dict, rewards, terminated, truncated, infos = env.step(actions)
        total_reward += sum(rewards.values())

        if any(terminated.values()) or any(truncated.values()):
            break

    env.scene.stop_recording(str(output_path))
    print(f"Recorded {step + 1} steps (reward: {total_reward:.2f}) → {output_path}")
    return output_path
