#!/usr/bin/env python3
"""Experiment 03: Growth

Coral creature learns to grow organically.
Growth tips emit new particles based on agent actions.
Reward favors interesting morphology over random expansion.

Run: python experiments/03_growth.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yeppoh.training import run_experiment

config = {
    "scene": {
        "n_envs": 128,
        "dt": 0.01,
        "max_steps": 2000,  # longer episodes for growth
        "gpu": True,
    },
    "creature": {
        "plan": "coral",
        "n_creatures": 1,
        "senses": ["proprioception", "echolocation", "chemoreception"],
        "params": {
            "stalk_height": 0.5,
            "n_branches": 4,
            "branch_length": 0.3,
        },
    },
    "brain": {
        "feature_dim": 128,
        "memory_dim": 64,
        "use_memory": True,
        "use_communication": True,
        "use_drives": True,
    },
    "reward": {
        "growth": 1.0,
        "coordination": 0.3,
        "survival": 0.5,
    },
    "training": {
        "algorithm": "mappo",
        "timesteps": 1_000_000,
        "output_dir": "runs/03_growth",
    },
    "world": {
        "pheromones": True,
        "pheromone_channels": 3,
        "energy": True,
        "stimuli": {
            "lights": [{"position": [0, 0, 5], "intensity": 1.0}],
        },
    },
}

if __name__ == "__main__":
    run_experiment(config)
