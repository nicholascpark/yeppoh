"""Training harness — experiment runner with algorithm selection."""

from .runner import run_experiment
from .algorithms import ALGORITHM_REGISTRY

__all__ = ["run_experiment", "ALGORITHM_REGISTRY"]
