"""Minimal PPO trainer for solo creature training.

Standard clipped-surrogate PPO with GAE advantages. Operates on
(T, n_envs) rollouts. Kept intentionally short — full PPO in ~120 lines.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class PPO:
    def __init__(
        self,
        policy,
        device: torch.device,
        lr: float = 3e-4,
        clip_eps: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        epochs: int = 4,
        minibatches: int = 4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        max_grad_norm: float = 0.5,
    ):
        self.policy = policy
        self.device = device
        self.optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.epochs = epochs
        self.minibatches = minibatches
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.max_grad_norm = max_grad_norm

    def compute_gae(
        self,
        rewards: np.ndarray,
        values: np.ndarray,
        dones: np.ndarray,
        last_value: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """GAE(lambda) advantage + return. Shapes: rewards/values/dones (T, n_envs), last_value (n_envs,)."""
        T, n_envs = rewards.shape
        advantages = np.zeros_like(rewards)
        gae = np.zeros(n_envs, dtype=np.float32)
        for t in reversed(range(T)):
            next_value = last_value if t == T - 1 else values[t + 1]
            next_nonterminal = 1.0 - dones[t].astype(np.float32)
            delta = rewards[t] + self.gamma * next_value * next_nonterminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * next_nonterminal * gae
            advantages[t] = gae
        returns = advantages + values
        return advantages, returns

    def update(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        log_probs_old: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
    ) -> dict[str, float]:
        """Run `epochs` PPO update passes over the flat batch."""
        n = obs.shape[0]
        batch_size = max(n // self.minibatches, 1)

        stats = {"pi_loss": [], "v_loss": [], "entropy": [], "approx_kl": []}

        for _ in range(self.epochs):
            perm = torch.randperm(n, device=obs.device)
            for start in range(0, n, batch_size):
                mb = perm[start:start + batch_size]
                mb_obs = obs[mb]
                mb_act = actions[mb]
                mb_lp_old = log_probs_old[mb]
                mb_adv = advantages[mb]
                mb_ret = returns[mb]

                # Per-minibatch advantage normalization stabilizes training
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                lp_new, v_new, entropy = self.policy.evaluate(mb_obs, mb_act)
                ratio = (lp_new - mb_lp_old).exp()
                clip = ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps)
                pi_loss = -(torch.min(ratio * mb_adv, clip * mb_adv)).mean()
                v_loss = F.mse_loss(v_new, mb_ret)
                ent = entropy.mean()

                loss = pi_loss + self.value_coef * v_loss - self.entropy_coef * ent

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (mb_lp_old - lp_new).mean().item()
                stats["pi_loss"].append(pi_loss.item())
                stats["v_loss"].append(v_loss.item())
                stats["entropy"].append(ent.item())
                stats["approx_kl"].append(approx_kl)

        return {k: float(np.mean(v)) for k, v in stats.items()}
