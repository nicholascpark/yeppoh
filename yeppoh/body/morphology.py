"""Body plan definitions — the shapes and layouts of creatures.

★ THIS IS WHERE YOU DESIGN NEW CREATURES.

Each body plan is a function that takes a Genesis scene and config,
adds entities to the scene, and returns a CreatureBody descriptor.

Register new plans by adding them to BODY_PLANS at the bottom.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import genesis as gs
import numpy as np

from .materials import make_material


@dataclass
class BodyPartSpec:
    """Describes one part of a creature before scene compilation."""

    name: str
    entity: Any = None  # set after scene.add_entity()
    material_name: str = "elastic"
    n_agents: int = 1  # how many RL agents control this part
    particle_range: tuple[int, int] = (0, 0)  # filled after build
    is_emitter: bool = False
    muscle_fiber_dir: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class BodyPlanResult:
    """Output of a body plan builder — everything needed to construct a CreatureBody."""

    parts: list[BodyPartSpec] = field(default_factory=list)
    constraints: list[dict] = field(default_factory=list)
    emitters: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


# ── Body Plans ────────────────────────────────────────────────────────


def build_blob(scene: Any, cfg: dict) -> BodyPlanResult:
    """Simple soft blob — the "hello world" creature.

    One sphere of muscle tissue. Good for testing locomotion via
    contraction/expansion cycles (jellyfish-like pulsing).
    """
    radius = cfg.get("radius", 0.3)
    n_agents = cfg.get("n_agents", 4)

    core = scene.add_entity(
        material=make_material("flesh"),
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, radius + 0.05),
            radius=radius,
        ),
    )

    result = BodyPlanResult(
        parts=[
            BodyPartSpec(
                name="core",
                entity=core,
                material_name="flesh",
                n_agents=n_agents,
            )
        ],
        metadata={"type": "blob", "radius": radius},
    )
    return result


def build_coral(scene: Any, cfg: dict) -> BodyPlanResult:
    """Branching coral creature.

    Central stalk with branches. Branches can grow via particle
    emission at tips. Good for testing growth + structural reward.
    """
    stalk_height = cfg.get("stalk_height", 0.5)
    n_branches = cfg.get("n_branches", 4)
    branch_length = cfg.get("branch_length", 0.3)

    # Central stalk — stiffer muscle
    stalk = scene.add_entity(
        material=make_material("elastic"),
        morph=gs.morphs.Cylinder(
            pos=(0.0, 0.0, stalk_height / 2 + 0.05),
            radius=0.08,
            height=stalk_height,
        ),
    )

    parts = [BodyPartSpec(name="stalk", entity=stalk, material_name="elastic")]
    constraints = []

    # Branches — softer, contractile
    for i in range(n_branches):
        angle = i * 2 * math.pi / n_branches
        bx = math.cos(angle) * 0.15
        by = math.sin(angle) * 0.15
        bz = stalk_height * 0.7

        branch = scene.add_entity(
            material=make_material("flesh"),
            morph=gs.morphs.Cylinder(
                pos=(bx, by, bz),
                radius=0.04,
                height=branch_length,
            ),
        )

        parts.append(BodyPartSpec(
            name=f"branch_{i}",
            entity=branch,
            material_name="flesh",
            muscle_fiber_dir=(math.cos(angle), math.sin(angle), 0.5),
        ))

        # Emitter at branch tip for growth
        tip_pos = (
            bx + math.cos(angle) * branch_length * 0.5,
            by + math.sin(angle) * branch_length * 0.5,
            bz + branch_length * 0.3,
        )
        emitter = scene.add_entity(
            material=make_material("slime"),
            morph=gs.morphs.Sphere(pos=tip_pos, radius=0.02),
            emitter=gs.options.EmitterOptions(
                max_particles=200,
                emission_rate=0,  # controlled by agent
            ),
        )
        parts.append(BodyPartSpec(
            name=f"tip_{i}",
            entity=emitter,
            material_name="slime",
            is_emitter=True,
        ))

    return BodyPlanResult(
        parts=parts,
        constraints=constraints,
        metadata={"type": "coral", "n_branches": n_branches},
    )


def build_jellyfish(scene: Any, cfg: dict) -> BodyPlanResult:
    """Jellyfish — bell + trailing tentacles.

    Bell contracts rhythmically for propulsion (FEM muscle).
    Tentacles are passive elastic strands that trail behind.
    """
    bell_radius = cfg.get("bell_radius", 0.25)
    n_tentacles = cfg.get("n_tentacles", 6)
    tentacle_length = cfg.get("tentacle_length", 0.4)

    # Bell — FEM membrane for clean contraction
    bell = scene.add_entity(
        material=make_material("membrane"),
        morph=gs.morphs.Sphere(
            pos=(0.0, 0.0, 0.8),
            radius=bell_radius,
        ),
    )

    parts = [BodyPartSpec(
        name="bell",
        entity=bell,
        material_name="membrane",
        n_agents=2,
        muscle_fiber_dir=(0.0, 0.0, -1.0),
    )]

    # Tentacles — soft elastic
    for i in range(n_tentacles):
        angle = i * 2 * math.pi / n_tentacles
        tx = math.cos(angle) * bell_radius * 0.6
        ty = math.sin(angle) * bell_radius * 0.6

        tentacle = scene.add_entity(
            material=make_material("slime"),
            morph=gs.morphs.Cylinder(
                pos=(tx, ty, 0.8 - tentacle_length / 2),
                radius=0.015,
                height=tentacle_length,
            ),
        )
        parts.append(BodyPartSpec(
            name=f"tentacle_{i}",
            entity=tentacle,
            material_name="slime",
        ))

    return BodyPlanResult(
        parts=parts,
        metadata={"type": "jellyfish", "bell_radius": bell_radius},
    )


# ── Registry ──────────────────────────────────────────────────────────
# ★ Add new body plans here. They become available via configs.

BODY_PLANS: dict[str, callable] = {
    "blob": build_blob,
    "coral": build_coral,
    "jellyfish": build_jellyfish,
}


def build_body(scene: Any, plan_name: str, cfg: dict) -> BodyPlanResult:
    """Build a creature body from a named plan."""
    if plan_name not in BODY_PLANS:
        raise ValueError(f"Unknown body plan '{plan_name}'. Available: {list(BODY_PLANS.keys())}")
    return BODY_PLANS[plan_name](scene, cfg)
