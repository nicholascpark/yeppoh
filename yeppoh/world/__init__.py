"""World layer — environmental systems external to Genesis physics.

These run on torch tensors alongside the Genesis simulation:
- Pheromone diffusion grid (chemical signaling)
- Energy / metabolism system
- Environmental stimuli (light, temperature)
- Reaction-diffusion patterns (Turing patterns on surfaces)
"""

from .pheromones import PheromoneGrid
from .energy import MetabolicSystem
from .stimuli import StimuliField
from .reaction_diffusion import ReactionDiffusion

__all__ = ["PheromoneGrid", "MetabolicSystem", "StimuliField", "ReactionDiffusion"]
