"""Train an RL agent to sculpt meshes.

Usage:
    python train.py                          # 500K steps, 4 parallel envs
    python train.py --timesteps 1000000      # longer training
    python train.py --resume checkpoints/sculpture_250000_steps.zip
"""

import argparse
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from sculpture.env import SculptureEnv


def make_env(rank: int = 0, seed: int = 0):
    def _init():
        env = SculptureEnv(
            subdivisions=2,
            n_anchors=16,
            max_steps=200,
            action_scale=0.02,
        )
        env.reset(seed=seed + rank)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train sculpture RL agent")
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--n-envs", type=int, default=4)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--log-dir", type=str, default="logs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Vectorized envs for parallel rollout collection
    if args.n_envs > 1:
        env = SubprocVecEnv([make_env(i, args.seed) for i in range(args.n_envs)])
    else:
        env = DummyVecEnv([make_env(0, args.seed)])

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(
                net_arch=dict(pi=[256, 256], vf=[256, 256]),
            ),
            verbose=1,
            tensorboard_log=str(log_dir),
            seed=args.seed,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=max(args.save_freq // args.n_envs, 1),
        save_path=str(output_dir),
        name_prefix="sculpture",
    )

    print(f"Training for {args.timesteps:,} timesteps with {args.n_envs} envs")
    print(f"Checkpoints → {output_dir}/")
    print(f"TensorBoard → tensorboard --logdir {log_dir}")

    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_cb,
        progress_bar=True,
    )

    final_path = output_dir / "sculpture_final"
    model.save(str(final_path))
    print(f"Done. Final model → {final_path}")

    env.close()


if __name__ == "__main__":
    main()
