#!/usr/bin/env python3
"""Train creatures.

Usage:
    python scripts/train.py
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py creature.plan=coral training.algorithm=ippo
    python scripts/train.py scene.n_envs=16 training.timesteps=10000
"""

import sys
from pathlib import Path

from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from yeppoh.training import run_experiment


def main():
    # Load base config
    base_path = Path(__file__).parent.parent / "configs" / "default.yaml"
    cfg = OmegaConf.load(base_path)

    # Apply CLI overrides
    cli_overrides = OmegaConf.from_cli(sys.argv[1:])

    # Check for --config flag
    if hasattr(cli_overrides, "config"):
        extra = OmegaConf.load(cli_overrides.config)
        cfg = OmegaConf.merge(cfg, extra)
        del cli_overrides["config"]

    cfg = OmegaConf.merge(cfg, cli_overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)

    run_experiment(cfg_dict)


if __name__ == "__main__":
    main()
