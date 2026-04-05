"""Cognitive architecture — the creature's brain.

The brain is a modular neural network that processes observations
and outputs actions. Components can be mixed and matched via config:

- encoder: sensory obs → feature vector
- memory: GRU temporal context
- communication: inter-agent message passing
- drives: internal motivational state
- policy: feature → action distribution
"""

from .encoder import SensoryEncoder
from .memory import TemporalMemory
from .communication import CommChannel
from .drives import DriveSystem
from .policy import CreatureBrain

__all__ = [
    "SensoryEncoder",
    "TemporalMemory",
    "CommChannel",
    "DriveSystem",
    "CreatureBrain",
]
