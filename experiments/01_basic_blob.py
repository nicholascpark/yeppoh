#!/usr/bin/env python3
"""Experiment 01: Basic Blob

The simplest creature — a soft blob learning to pulse.
Validates that the env, physics, and training loop work end-to-end.

Run: python experiments/01_basic_blob.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from yeppoh.training import run_experiment

config = {
    "scene": {
        "n_envs": 4,
        "dt": 0.01,
        "substeps": 32,  # substep_dt = dt/substeps ≈ 3e-4, matches Genesis stability suggestion
        "max_steps": 500,
        "gpu": True,
        "show_viewer": False,
    },
    "creature": {
        "plan": "blob",
        "n_creatures": 1,
        "senses": ["proprioception"],  # minimal senses
        "sense_params": {},
        "params": {"radius": 0.3, "n_agents": 2},  # just 2 agents
    },
    "brain": {
        "feature_dim": 64,  # smaller brain
        "hidden_dim": 128,
        "memory_dim": 32,
        "use_memory": True,
        "use_communication": False,  # no comms with 2 agents
        "use_drives": False,
    },
    "reward": {
        "survival": 1.0,
    },
    "training": {
        "algorithm": "mappo",
        "timesteps": 100_000,
        "output_dir": "runs/01_basic_blob",
        "hyperparams": {"lr": 3e-4, "n_steps": 1024},
    },
    "world": {
        "pheromones": False,
        "energy": False,
    },
}

if __name__ == "__main__":
    run_experiment(config)
