"""Material catalog for creature body parts.

Each material maps to a Genesis MPM or FEM material with tuned physical
properties. Add new materials here when you want new tissue types.

Genesis material params are compile-time constants — they cannot change
mid-simulation. Use MPM.Muscle actuation for runtime stiffness control.
"""

from dataclasses import dataclass
from typing import Any

import genesis as gs


@dataclass
class MaterialSpec:
    """Describes a material before Genesis scene compilation."""

    name: str
    solver: str  # "mpm" or "fem"
    genesis_cls: str  # e.g. "MPM.Muscle", "MPM.Elastic"
    params: dict[str, Any]
    description: str = ""


# ── Material Catalog ──────────────────────────────────────────────────
# ★ Add new materials here. They become available in morphology configs.

MATERIAL_SPECS: dict[str, MaterialSpec] = {
    "flesh": MaterialSpec(
        name="flesh",
        solver="mpm",
        genesis_cls="MPM.Muscle",
        params=dict(E=5e3, nu=0.45, rho=1000.0),
        description="Soft contractile tissue. Responds to actuation signals.",
    ),
    "bone": MaterialSpec(
        name="bone",
        solver="mpm",
        genesis_cls="MPM.Elastic",
        params=dict(E=1e6, nu=0.3, rho=2000.0),
        description="Rigid structural material. High Young's modulus.",
    ),
    "elastic": MaterialSpec(
        name="elastic",
        solver="mpm",
        genesis_cls="MPM.Elastic",
        params=dict(E=1e4, nu=0.4, rho=1000.0),
        description="General-purpose elastic tissue.",
    ),
    "slime": MaterialSpec(
        name="slime",
        solver="mpm",
        genesis_cls="MPM.Elastic",
        params=dict(E=500.0, nu=0.48, rho=800.0),
        description="Very soft, nearly incompressible goo.",
    ),
    "membrane": MaterialSpec(
        name="membrane",
        solver="fem",
        genesis_cls="FEM.Muscle",
        params=dict(E=2e3, nu=0.4, rho=500.0),
        description="Thin elastic membrane. FEM for better surface behavior.",
    ),
    "fluid": MaterialSpec(
        name="fluid",
        solver="mpm",
        genesis_cls="MPM.Liquid",
        params=dict(E=1e4, rho=1000.0, viscosity=0.01),
        description="Internal fluid for hydraulic actuation.",
    ),
}


def make_material(name: str) -> Any:
    """Instantiate a Genesis material from the catalog."""
    spec = MATERIAL_SPECS[name]
    # Navigate to the Genesis material class
    parts = spec.genesis_cls.split(".")
    cls = gs.materials
    for part in parts:
        cls = getattr(cls, part)
    return cls(**spec.params)


# Convenience: pre-resolved dict for quick access
MATERIALS = {name: spec for name, spec in MATERIAL_SPECS.items()}
