#!/usr/bin/env python3
"""Experiment 02: Locomotion

Blob learns to crawl forward via coordinated contraction.
This is where multi-agent coordination matters —
agents must pulse in sync to produce coherent movement.

Run: python experiments/02_locomotion.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yeppoh.training import run_experiment

config = {
    "scene": {
        "n_envs": 256,
        "dt": 0.01,
        "max_steps": 1000,
        "gpu": True,
    },
    "creature": {
        "plan": "blob",
        "n_creatures": 1,
        "senses": ["proprioception", "touch"],
        "params": {"radius": 0.3, "n_agents": 4},
    },
    "brain": {
        "feature_dim": 128,
        "memory_dim": 64,
        "use_memory": True,
        "use_communication": True,
        "use_drives": False,
    },
    "reward": {
        "locomotion": 1.0,
        "coordination": 0.5,
        "survival": 0.3,
    },
    "training": {
        "algorithm": "mappo",
        "timesteps": 500_000,
        "output_dir": "runs/02_locomotion",
    },
    "world": {
        "pheromones": False,
        "energy": True,
    },
}

if __name__ == "__main__":
    run_experiment(config)
