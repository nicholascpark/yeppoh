#!/usr/bin/env python3
"""Phylum I sculpture training — solo single-agent loop.

One creature, one policy, one reward. Fully vectorized across Genesis
envs (every env contributes to the gradient, unlike the MARL path).

This is the Phase A smoke test: prove PPO updates the brain and a blob
measurably learns. Once that works, swap `reward` to a Phase B candidate
(rhythmic / stimulus / eventually the learned aesthetic reward per
DIRECTION.md §10) and run curation pools.

Local (Mac mini, debugging):
    python experiments/phylum1_sculpt.py

Cloud (Modal, real training):
    modal run scripts/modal_train.py --experiment phylum1_sculpt
    # then raise n_envs, timesteps in this config.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yeppoh.solo import run_solo


# ── Config ────────────────────────────────────────────────────────────
# ★ Tune these. Reward, n_envs, timesteps are the most-used knobs.

config = {
    "scene": {
        # ★ This config targets cloud A100 via Modal. For a local MBA smoke test,
        # drop n_envs=4, max_steps=64, n_steps=64, timesteps=1024, checkpoint_every=2.
        "n_envs": 32,             # Cloud A100. Local MBA: drop to 4.
        "dt": 0.01,
        "substeps": 32,           # substep_dt ≈ 3e-4, matches Genesis stability
        "max_steps": 200,         # episode length in control steps (~2s simulated)
        "gpu": True,
        "show_viewer": False,
    },
    "creature": {
        "plan": "blob",
        "params": {"radius": 0.3},
    },
    "brain": {
        "hidden": 64,
    },
    "reward": "locomotion",       # "locomotion" | "none" (Phase B: "rhythmic", "stimulus", ...)
    "training": {
        "lr": 3e-4,
        "n_steps": 256,           # steps per iteration per env
        "timesteps": 50_000,      # First cloud run — ~15 min on A100, ~$1.
                                  # Bump to 200_000+ once learning is verified.
        "epochs": 4,
        "minibatches": 4,
        "clip_eps": 0.2,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "entropy_coef": 0.01,
        "checkpoint_every": 5,
        "output_dir": "runs/phylum1_sculpt",
    },
}


if __name__ == "__main__":
    run_solo(config)
