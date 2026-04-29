#!/usr/bin/env python3
"""Visualize a solo training checkpoint.

Loads a `.pt` saved by `yeppoh.solo.run_solo`, prints the training-history
table with a learning-trend summary, then opens the Genesis viewer and
plays the trained policy back deterministically so you can watch what
the creature learned.

Usage:
    python scripts/visualize_solo.py runs/phylum1_sculpt/final.pt
    python scripts/visualize_solo.py runs/phylum1_sculpt/final.pt --steps 300
    python scripts/visualize_solo.py runs/phylum1_sculpt/final.pt --no-viewer
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from yeppoh.solo import SoloYeppohEnv, SoloPolicy


def print_history(history: dict) -> None:
    n = len(history.get("mean_reward", []))
    if n == 0:
        print("(no training history in checkpoint)")
        return

    print(f"\n=== Training history ({n} iterations) ===")
    print(f"{'it':>3} | {'rew/step':>10} | {'pi_loss':>9} | {'v_loss':>9} | {'entropy':>8} | {'kl':>7}")
    for i in range(n):
        print(
            f"{i+1:>3} | "
            f"{history['mean_reward'][i]:>+10.5f} | "
            f"{history['pi_loss'][i]:>+9.4f} | "
            f"{history['v_loss'][i]:>9.5f} | "
            f"{history['entropy'][i]:>+8.3f} | "
            f"{history['approx_kl'][i]:>+7.3f}"
        )

    # Trend summary — is the reward going up?
    if n >= 4:
        q = max(1, n // 4)
        rew_first = float(np.mean(history["mean_reward"][:q]))
        rew_last = float(np.mean(history["mean_reward"][-q:]))
        ent_first = float(np.mean(history["entropy"][:q]))
        ent_last = float(np.mean(history["entropy"][-q:]))
        v_first = float(np.mean(history["v_loss"][:q]))
        v_last = float(np.mean(history["v_loss"][-q:]))

        print()
        print(f"Reward     first→last quartile:  {rew_first:+.5f}  →  {rew_last:+.5f}")
        print(f"Entropy    first→last quartile:  {ent_first:+.3f}    →  {ent_last:+.3f}     (lower = more decisive policy)")
        print(f"Value loss first→last quartile:  {v_first:.5f}   →  {v_last:.5f}    (lower = critic fitting reward better)")

        rew_up = rew_last > rew_first + 0.0005
        v_down = v_last < v_first * 0.5
        ent_down = ent_last < ent_first - 0.05

        print()
        if rew_up and ent_down:
            print("→ Policy is learning the task. Reward up, entropy down.")
        elif rew_up:
            print("→ Reward is trending up but entropy is unchanged. Early stage of learning.")
        elif v_down and not rew_up:
            print("→ Critic is converging but reward is flat. Either too few samples to see policy improvement, or no exploitable signal in this reward.")
        else:
            print("→ No clear learning signal yet. Probably need more iterations / cloud training.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint from yeppoh.solo")
    parser.add_argument("--steps", type=int, default=300, help="Playback steps in viewer")
    parser.add_argument("--no-viewer", action="store_true", help="Skip Genesis viewer (just print history)")
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    if "cfg" not in ckpt or "policy" not in ckpt:
        sys.exit(f"Checkpoint at {args.checkpoint} doesn't look like a solo checkpoint (missing 'cfg' or 'policy'). Use scripts/visualize.py for MARL checkpoints.")

    cfg = ckpt["cfg"]
    print_history(ckpt.get("history", {}))

    if args.no_viewer:
        return

    print("\n=== Playing back trained policy ===")
    print("Opening Genesis viewer. Close the window when done watching.")

    # Force single env for playback so the viewer renders cleanly
    cfg = dict(cfg)
    cfg["scene"] = {**cfg["scene"], "n_envs": 1, "show_viewer": True}

    env = SoloYeppohEnv(cfg)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    policy = SoloPolicy(
        env.obs_dim,
        env.act_dim,
        hidden=cfg.get("brain", {}).get("hidden", 64),
    ).to(device)
    policy.load_state_dict(ckpt["policy"])
    policy.eval()

    obs = env.reset()
    total_reward = 0.0
    for step in range(args.steps):
        obs_t = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            mean, _, _ = policy.forward(obs_t)
            # Deterministic playback — use the policy mean, not a sample
            action = mean.cpu().numpy()
        obs, reward, done, _ = env.step(action)
        total_reward += float(reward[0])
        if done.any():
            obs = env.reset()

    print(f"\nPlayback total reward over {args.steps} steps: {total_reward:+.4f}")
    env.close()


if __name__ == "__main__":
    main()
