#!/usr/bin/env python3
"""Visualize a trained creature.

Usage:
    python scripts/visualize.py runs/default/final.pt
    python scripts/visualize.py runs/default/final.pt --record video.mp4
    python scripts/visualize.py runs/default/final.pt --export-mesh output.glb
"""

import sys
import argparse
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yeppoh.env import YeppohEnv
from yeppoh.brain import CreatureBrain
from yeppoh.export.mesh import export_mesh
from yeppoh.export.video import record_episode


def main():
    parser = argparse.ArgumentParser(description="Visualize trained creature")
    parser.add_argument("checkpoint", help="Path to checkpoint .pt file")
    parser.add_argument("--record", help="Record episode to MP4")
    parser.add_argument("--export-mesh", help="Export final mesh as GLB")
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]

    # Enable viewer for visualization
    cfg["scene"]["show_viewer"] = True
    cfg["scene"]["n_envs"] = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Rebuild env and brain
    env = YeppohEnv(cfg)

    brain_cfg = cfg.get("brain", {})
    brain = CreatureBrain(
        obs_dim=env._obs_dim,
        action_dim=env._act_dim,
        feature_dim=brain_cfg.get("feature_dim", 128),
        memory_dim=brain_cfg.get("memory_dim", 64),
        message_dim=brain_cfg.get("message_dim", 16),
        use_memory=brain_cfg.get("use_memory", True),
        use_communication=brain_cfg.get("use_communication", True),
        use_drives=brain_cfg.get("use_drives", True),
    ).to(device)

    brain.load_state_dict(ckpt["brain"])
    brain.eval()

    if args.record:
        record_episode(env, brain, args.record, max_steps=args.steps, device=str(device))
    else:
        # Interactive visualization
        obs_dict, _ = env.reset()
        for step in range(args.steps):
            actions = {}
            for agent_id, obs in obs_dict.items():
                obs_t = torch.tensor(obs, device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    action, _ = brain.get_action(obs_t)
                actions[agent_id] = action.squeeze(0).cpu().numpy()
            obs_dict, rewards, terminated, truncated, infos = env.step(actions)
            if any(terminated.values()) or any(truncated.values()):
                obs_dict, _ = env.reset()

    env.close()


if __name__ == "__main__":
    main()
