"""Solo training orchestrator — builds env/policy/PPO, runs the loop."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .env import SoloYeppohEnv
from .policy import SoloPolicy
from .ppo import PPO


def _select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_solo(cfg: dict) -> dict[str, Any]:
    """Train one creature with PPO. Returns history dict."""
    print("═══ YEPPOH SOLO ═══")
    print(f"Creature: {cfg.get('creature', {}).get('plan', 'blob')}")
    print(f"Reward:   {cfg.get('reward', 'locomotion')}")

    env = SoloYeppohEnv(cfg)
    device = _select_device()
    print(f"Device:   {device}")

    brain_cfg = cfg.get("brain", {})
    policy = SoloPolicy(env.obs_dim, env.act_dim, hidden=brain_cfg.get("hidden", 64)).to(device)

    train_cfg = cfg.get("training", {})
    ppo = PPO(
        policy,
        device,
        lr=train_cfg.get("lr", 3e-4),
        clip_eps=train_cfg.get("clip_eps", 0.2),
        value_coef=train_cfg.get("value_coef", 0.5),
        entropy_coef=train_cfg.get("entropy_coef", 0.01),
        epochs=train_cfg.get("epochs", 4),
        minibatches=train_cfg.get("minibatches", 4),
        gamma=train_cfg.get("gamma", 0.99),
        gae_lambda=train_cfg.get("gae_lambda", 0.95),
        max_grad_norm=train_cfg.get("max_grad_norm", 0.5),
    )

    n_steps = train_cfg.get("n_steps", 128)
    total_ts = train_cfg.get("timesteps", 50_000)
    n_iter = max(total_ts // (n_steps * env.n_envs), 1)
    checkpoint_every = train_cfg.get("checkpoint_every", 10)

    output_dir = Path(train_cfg.get("output_dir", "runs/solo"))
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"n_envs={env.n_envs}, obs_dim={env.obs_dim}, act_dim={env.act_dim}")
    print(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")
    print(f"Training: {total_ts:,} timesteps × {n_iter} iterations, {n_steps} steps/iter")
    print(f"Output:   {output_dir}\n")

    # Rollout storage
    T, N = n_steps, env.n_envs
    obs_buf = np.zeros((T, N, env.obs_dim), dtype=np.float32)
    act_buf = np.zeros((T, N, env.act_dim), dtype=np.float32)
    lp_buf = np.zeros((T, N), dtype=np.float32)
    val_buf = np.zeros((T, N), dtype=np.float32)
    rew_buf = np.zeros((T, N), dtype=np.float32)
    done_buf = np.zeros((T, N), dtype=np.float32)

    history: dict[str, list[float]] = {"mean_reward": [], "pi_loss": [], "v_loss": [], "entropy": [], "approx_kl": []}

    obs = env.reset()

    for it in range(n_iter):
        # Collect rollout
        for t in range(T):
            obs_t = torch.from_numpy(obs).to(device)
            with torch.no_grad():
                action_t, lp, val = policy.act(obs_t)

            obs_buf[t] = obs
            act_buf[t] = action_t.cpu().numpy()
            lp_buf[t] = lp.cpu().numpy()
            val_buf[t] = val.cpu().numpy()

            next_obs, rew, done, _ = env.step(act_buf[t])
            rew_buf[t] = rew
            done_buf[t] = done.astype(np.float32)

            # Genesis resets all envs together at max_steps — match that
            if done.any():
                next_obs = env.reset()
            obs = next_obs

        # Bootstrap with value of final obs
        obs_t = torch.from_numpy(obs).to(device)
        with torch.no_grad():
            _, _, last_val = policy.forward(obs_t)
        last_val_np = last_val.cpu().numpy()

        advantages, returns = ppo.compute_gae(rew_buf, val_buf, done_buf, last_val_np)

        obs_flat = torch.from_numpy(obs_buf.reshape(-1, env.obs_dim)).to(device)
        act_flat = torch.from_numpy(act_buf.reshape(-1, env.act_dim)).to(device)
        lp_flat = torch.from_numpy(lp_buf.reshape(-1)).to(device)
        adv_flat = torch.from_numpy(advantages.reshape(-1)).to(device)
        ret_flat = torch.from_numpy(returns.reshape(-1)).to(device)

        stats = ppo.update(obs_flat, act_flat, lp_flat, adv_flat, ret_flat)

        mean_reward_per_env_per_step = float(rew_buf.mean())
        history["mean_reward"].append(mean_reward_per_env_per_step)
        for k in ("pi_loss", "v_loss", "entropy", "approx_kl"):
            history[k].append(stats[k])

        print(
            f"it {it+1:3d}/{n_iter}  "
            f"rew/step {mean_reward_per_env_per_step:+.4f}  "
            f"π {stats['pi_loss']:+.3f}  "
            f"v {stats['v_loss']:.3f}  "
            f"ent {stats['entropy']:+.2f}  "
            f"kl {stats['approx_kl']:.3f}"
        )

        if (it + 1) % checkpoint_every == 0:
            torch.save(
                {"policy": policy.state_dict(), "cfg": cfg, "iteration": it + 1, "history": history},
                output_dir / f"ckpt_{it+1:04d}.pt",
            )

    torch.save(
        {"policy": policy.state_dict(), "cfg": cfg, "history": history},
        output_dir / "final.pt",
    )
    env.close()
    return {"history": history, "iterations": n_iter}
