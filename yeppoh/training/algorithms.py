"""Algorithm registry — TorchRL/BenchMARL algorithm configs.

★ This is where you add new RL algorithms or tune hyperparameters.

Each algorithm entry maps a name to a setup function that returns
a configured TorchRL loss module and optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class AlgorithmConfig:
    """Configuration for one MARL algorithm."""

    name: str
    description: str
    # TorchRL loss class name
    loss_class: str
    # Default hyperparameters
    defaults: dict[str, Any]


# ── Algorithm Catalog ─────────────────────────────────────────────────

ALGORITHM_REGISTRY: dict[str, AlgorithmConfig] = {
    "mappo": AlgorithmConfig(
        name="mappo",
        description="Multi-Agent PPO with centralized value function (CTDE)",
        loss_class="torchrl.objectives.ClipPPOLoss",
        defaults=dict(
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            value_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=10,
            batch_size=64,
            n_steps=2048,
        ),
    ),
    "ippo": AlgorithmConfig(
        name="ippo",
        description="Independent PPO — each agent learns separately",
        loss_class="torchrl.objectives.ClipPPOLoss",
        defaults=dict(
            lr=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_epsilon=0.2,
            entropy_coef=0.02,  # more exploration for independent learners
            value_coef=0.5,
            max_grad_norm=0.5,
            n_epochs=10,
            batch_size=64,
            n_steps=2048,
            shared_critic=False,
        ),
    ),
    "maddpg": AlgorithmConfig(
        name="maddpg",
        description="Multi-Agent DDPG — off-policy, continuous actions",
        loss_class="torchrl.objectives.DDPGLoss",
        defaults=dict(
            lr_actor=1e-4,
            lr_critic=3e-4,
            gamma=0.99,
            tau=0.005,
            batch_size=256,
            buffer_size=100_000,
            noise_std=0.1,
        ),
    ),
    "qmix": AlgorithmConfig(
        name="qmix",
        description="QMIX — value decomposition for cooperative agents",
        loss_class="torchrl.objectives.QMixerLoss",
        defaults=dict(
            lr=5e-4,
            gamma=0.99,
            batch_size=32,
            buffer_size=50_000,
            target_update_freq=200,
            mixing_embed_dim=32,
        ),
    ),
}


def get_algorithm(name: str, overrides: dict | None = None) -> AlgorithmConfig:
    """Get algorithm config with optional hyperparameter overrides."""
    if name not in ALGORITHM_REGISTRY:
        available = list(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"Unknown algorithm '{name}'. Available: {available}")

    config = ALGORITHM_REGISTRY[name]
    if overrides:
        merged = {**config.defaults, **overrides}
        config = AlgorithmConfig(
            name=config.name,
            description=config.description,
            loss_class=config.loss_class,
            defaults=merged,
        )
    return config
