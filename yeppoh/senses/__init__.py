"""Sensory layer — how creatures perceive the world.

Each sense module reads from Genesis sensors or world state and
returns a flat tensor. The SensorySystem composes all active senses
into a single observation vector per agent.

★ To add a new sense: create a new SenseModule subclass, register it below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch

from .echolocation import Echolocation
from .touch import TouchSense
from .vision import Vision
from .proprioception import Proprioception
from .chemoreception import Chemoreception


class SenseModule(ABC):
    """Base class for all sensory modules."""

    name: str = "unnamed"
    obs_dim: int = 0

    @abstractmethod
    def setup(self, scene: Any, entity: Any, cfg: dict) -> None:
        """Attach Genesis sensors to entities."""

    @abstractmethod
    def read(self, env_idx: int | None = None) -> torch.Tensor:
        """Read sensor data. Returns (n_envs, obs_dim)."""

    def reset(self) -> None:
        """Called on episode reset."""


# ── Sense Registry ────────────────────────────────────────────────────
# ★ Add new senses here. They become available via creature config.

SENSE_REGISTRY: dict[str, type[SenseModule]] = {
    "proprioception": Proprioception,
    "echolocation": Echolocation,
    "touch": TouchSense,
    "vision": Vision,
    "chemoreception": Chemoreception,
}


@dataclass
class SensoryReading:
    """All sensor readings for one agent, one timestep."""

    tensors: dict[str, torch.Tensor]  # sense_name → (n_envs, dim)

    @property
    def flat(self) -> torch.Tensor:
        """Concatenate all readings into one vector. (n_envs, total_dim)."""
        if not self.tensors:
            return torch.zeros(1, 0)
        return torch.cat(list(self.tensors.values()), dim=-1)

    @property
    def total_dim(self) -> int:
        return sum(t.shape[-1] for t in self.tensors.values())


class SensorySystem:
    """Composes multiple senses into a unified observation per agent."""

    def __init__(self, sense_names: list[str], cfg: dict):
        self.modules: dict[str, SenseModule] = {}
        for name in sense_names:
            if name not in SENSE_REGISTRY:
                raise ValueError(
                    f"Unknown sense '{name}'. Available: {list(SENSE_REGISTRY.keys())}"
                )
            self.modules[name] = SENSE_REGISTRY[name]()

        self.cfg = cfg

    def setup(self, scene: Any, entity: Any) -> None:
        for mod in self.modules.values():
            mod.setup(scene, entity, self.cfg)

    def read(self, env_idx: int | None = None) -> SensoryReading:
        tensors = {}
        for name, mod in self.modules.items():
            tensors[name] = mod.read(env_idx)
        return SensoryReading(tensors=tensors)

    def reset(self) -> None:
        for mod in self.modules.values():
            mod.reset()

    @property
    def total_obs_dim(self) -> int:
        return sum(m.obs_dim for m in self.modules.values())


__all__ = [
    "SenseModule",
    "SensorySystem",
    "SensoryReading",
    "SENSE_REGISTRY",
]
