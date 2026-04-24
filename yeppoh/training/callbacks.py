"""Training callbacks — logging, checkpointing, visualization."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class CallbackManager:
    """Manages training callbacks for logging and monitoring."""

    def __init__(self, output_dir: Path, cfg: dict):
        self.output_dir = output_dir
        self.cfg = cfg
        self.writer = None

        # Try to set up TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=str(output_dir / "tb"))
        except ImportError:
            pass

    def on_iteration(self, iteration: int, results: dict[str, Any]) -> None:
        rewards = results.get("rewards", [])

        if rewards and self.writer:
            recent = rewards[-10:] if len(rewards) >= 10 else rewards
            self.writer.add_scalar("reward/mean", np.mean(recent), iteration)
            self.writer.add_scalar("reward/max", np.max(recent), iteration)

        # Progress log
        if (iteration + 1) % 10 == 0:
            recent = rewards[-10:] if len(rewards) >= 10 else rewards
            mean_r = np.mean(recent) if recent else 0.0
            print(f"  iter {iteration + 1:>5d} | reward: {mean_r:.3f}")

    def close(self) -> None:
        if self.writer:
            self.writer.close()
