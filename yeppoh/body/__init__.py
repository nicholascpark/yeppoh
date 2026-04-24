"""Body layer — Genesis physics entities that make up a creature."""

from .creature import CreatureBody
from .materials import MATERIALS, make_material
from .morphology import BODY_PLANS, build_body
from .actuators import ActuatorInterface

__all__ = [
    "CreatureBody",
    "MATERIALS",
    "make_material",
    "BODY_PLANS",
    "build_body",
    "ActuatorInterface",
]
