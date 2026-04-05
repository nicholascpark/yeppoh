"""Genesis scene builder.

Sets up the Genesis simulation scene with:
- Ground plane
- Creature bodies
- Sensors
- World systems (pheromones, energy, stimuli)

This is the glue between configs and the Genesis API.
"""

from __future__ import annotations

from typing import Any

import genesis as gs

from ..body import build_body, CreatureBody
from ..body.morphology import BodyPlanResult
from ..senses import SensorySystem
from ..world import PheromoneGrid, MetabolicSystem, StimuliField


def build_scene(cfg: dict) -> dict[str, Any]:
    """Build a complete Genesis scene from config.

    Returns a dict with all components needed by the environment:
    - scene: the Genesis Scene object
    - creatures: list of CreatureBody
    - senses: dict of agent_id → SensorySystem
    - world: dict with pheromone grid, energy, stimuli
    """
    scene_cfg = cfg.get("scene", {})
    creature_cfg = cfg.get("creature", {})
    world_cfg = cfg.get("world", {})

    # Initialize Genesis
    gs.init(backend=gs.cpu if not scene_cfg.get("gpu", True) else gs.gpu)

    # Create scene
    dt = scene_cfg.get("dt", 0.01)
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=dt,
            substeps=scene_cfg.get("substeps", 2),
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
        ),
        show_viewer=scene_cfg.get("show_viewer", False),
    )

    # Ground plane
    scene.add_entity(gs.morphs.Plane())

    # Build creatures
    n_creatures = creature_cfg.get("n_creatures", 1)
    plan_name = creature_cfg.get("plan", "blob")
    plan_cfg = creature_cfg.get("params", {})

    creatures: list[CreatureBody] = []
    for i in range(n_creatures):
        body_plan = build_body(scene, plan_name, plan_cfg)
        creature = CreatureBody(body_plan, creature_id=i, dt=dt)
        creatures.append(creature)

    # Build the scene (compiles simulation)
    n_envs = scene_cfg.get("n_envs", 256)
    scene.build(n_envs=n_envs)

    # Finalize creatures (resolve particle indices)
    for creature in creatures:
        creature.finalize(n_envs)

    # Set up sensory systems per agent
    sense_names = creature_cfg.get("senses", ["proprioception", "echolocation", "touch"])
    sense_cfg = creature_cfg.get("sense_params", {})
    senses: dict[str, SensorySystem] = {}
    for creature in creatures:
        for cluster in creature.clusters:
            sys = SensorySystem(sense_names, sense_cfg)
            sys.setup(scene, cluster.entity)
            senses[cluster.agent_id] = sys

    # World systems
    device = "cuda" if scene_cfg.get("gpu", True) else "cpu"
    total_agents = sum(c.n_agents for c in creatures)

    pheromone_grid = PheromoneGrid(
        n_channels=world_cfg.get("pheromone_channels", 3),
        grid_size=world_cfg.get("pheromone_grid_size", 64),
        device=device,
    ) if world_cfg.get("pheromones", True) else None

    energy_system = MetabolicSystem(
        n_agents=total_agents,
        n_envs=n_envs,
        device=device,
    ) if world_cfg.get("energy", True) else None

    stimuli = StimuliField(
        world_cfg.get("stimuli", {}),
        device=device,
    )

    return {
        "scene": scene,
        "creatures": creatures,
        "senses": senses,
        "pheromone_grid": pheromone_grid,
        "energy_system": energy_system,
        "stimuli": stimuli,
        "n_envs": n_envs,
        "dt": dt,
    }
