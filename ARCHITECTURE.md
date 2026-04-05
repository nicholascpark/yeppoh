# Architecture

## The 30-Second Picture

```
╔══════════════════════════════════════════════════════════════════════════╗
║                          YEPPOH SYSTEM                                  ║
║                                                                         ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                    configs/*.yaml                               │    ║
║  │              (mix & match experiment knobs)                      │    ║
║  └──────────────────────────┬──────────────────────────────────────┘    ║
║                             │                                           ║
║                             ▼                                           ║
║  ┌─────────────────────────────────────────────────────────────────┐    ║
║  │                   scripts/train.py                              │    ║
║  │               (CLI entry point — just runs configs)             │    ║
║  └──────────────────────────┬──────────────────────────────────────┘    ║
║                             │                                           ║
║       ┌─────────────────────┼─────────────────────┐                    ║
║       ▼                     ▼                     ▼                    ║
║  ┌─────────┐    ┌───────────────────┐    ┌──────────────┐              ║
║  │TRAINING │    │   ENVIRONMENT     │    │   EXPORT     │              ║
║  │         │    │                   │    │              │              ║
║  │ TorchRL │◄──►│  PettingZoo       │    │ mesh → glTF  │              ║
║  │BenchMARL│    │  multi-agent env  │    │ video → mp4  │              ║
║  │         │    │                   │    │ scene → WebXR│              ║
║  │ MAPPO   │    └────────┬──────────┘    └──────────────┘              ║
║  │ QMIX    │             │                                             ║
║  │ IPPO    │    ┌────────┴──────────┐                                  ║
║  │ MADDPG  │    │    Per-Agent      │                                  ║
║  └─────────┘    │  ┌──────┐ ┌─────┐│                                  ║
║                 │  │ obs  │ │ act ││                                  ║
║                 │  └──┬───┘ └──▲──┘│                                  ║
║                 └─────┼────────┼───┘                                  ║
║                       │        │                                       ║
║       ┌───────────────┼────────┼───────────────────┐                   ║
║       │               ▼        │                   │                   ║
║       │  ┌──────────────────────────────────────┐  │                   ║
║       │  │            BRAIN                      │  │                   ║
║       │  │                                       │  │                   ║
║       │  │  senses → encoder → memory → policy   │  │                   ║
║       │  │                       ↕                │  │                   ║
║       │  │              communication ←→ others   │  │                   ║
║       │  │                       ↕                │  │                   ║
║       │  │                    drives              │  │                   ║
║       │  └──────────────────────────────────────┘  │                   ║
║       │              AGENT (x N per creature)      │                   ║
║       └───────────────┬────────┬───────────────────┘                   ║
║                       │        │                                       ║
║       ┌───────────────┼────────┼───────────────────┐                   ║
║       │               ▼        │                   │                   ║
║       │  ┌─────────┐  ┌───────┴──┐  ┌──────────┐  │                   ║
║       │  │ SENSES  │  │   BODY   │  │  WORLD   │  │                   ║
║       │  │         │  │          │  │          │  │                   ║
║       │  │ echo    │  │ Genesis  │  │pheromone │  │                   ║
║       │  │ touch   │  │ MPM/FEM  │  │ energy   │  │                   ║
║       │  │ vision  │  │ muscles  │  │ stimuli  │  │                   ║
║       │  │ chemo   │  │ growth   │  │ R-D      │  │                   ║
║       │  │ proprio │  │          │  │          │  │                   ║
║       │  └─────────┘  └──────────┘  └──────────┘  │                   ║
║       │              GENESIS SCENE                  │                   ║
║       └─────────────────────────────────────────────┘                   ║
╚══════════════════════════════════════════════════════════════════════════╝
```

---

## Where You Experiment

| I want to...                          | Edit this                        | Config key          |
|---------------------------------------|----------------------------------|---------------------|
| **Design a new creature body**        | `yeppoh/body/morphology.py`      | `configs/creature/` |
| **Add a new material (slime, shell)** | `yeppoh/body/materials.py`       | `creature.materials` |
| **Add a new sense (magnetism, IR)**   | `yeppoh/senses/` (new file)      | `creature.senses`   |
| **Change how the brain works**        | `yeppoh/brain/`                  | `configs/brain/`    |
| **Add a communication protocol**      | `yeppoh/brain/communication.py`  | `brain.comm`        |
| **Design a new reward function**      | `yeppoh/env/rewards.py`          | `configs/reward/`   |
| **Try a different RL algorithm**      | `configs/training/*.yaml`        | `training.algorithm` |
| **Add world features (food, walls)**  | `yeppoh/world/stimuli.py`        | `world.stimuli`     |
| **Change pheromone behavior**         | `yeppoh/world/pheromones.py`     | `world.pheromones`  |
| **Run a quick experiment**            | `experiments/` (new script)      | —                   |
| **Export to VR**                      | `yeppoh/export/gallery.py`       | —                   |

---

## File Tree

```
yeppoh/
│
├── ARCHITECTURE.md              ← you are here
├── README.md                    ← project overview + quick start
├── pyproject.toml               ← package config
├── requirements.txt             ← dependencies
│
│   ════════════════════════════════════════════════════
│   EXPERIMENT SURFACES — where you spend most of your time
│   ════════════════════════════════════════════════════
│
├── configs/                     ★ MIX & MATCH EXPERIMENT KNOBS
│   ├── default.yaml             │  base config (all defaults)
│   ├── creature/                │
│   │   ├── blob.yaml            │  simple soft blob
│   │   ├── coral.yaml           │  branching coral creature
│   │   └── jellyfish.yaml       │  bell + tentacles
│   ├── brain/                   │
│   │   ├── reactive.yaml        │  no memory, pure reflex
│   │   ├── memory.yaml          │  GRU memory, no comms
│   │   └── social.yaml          │  memory + communication + drives
│   ├── reward/                  │
│   │   ├── locomotion.yaml      │  move forward
│   │   ├── growth.yaml          │  grow interestingly
│   │   └── survive.yaml         │  stay alive, find energy
│   └── training/                │
│       ├── mappo.yaml           │  centralized critic (default)
│       ├── ippo.yaml            │  independent learners
│       └── qmix.yaml            │  value decomposition
│
├── experiments/                 ★ RUNNABLE EXPERIMENT SCRIPTS
│   ├── 01_basic_blob.py         │  simplest creature, learn to pulse
│   ├── 02_locomotion.py         │  blob learns to crawl
│   ├── 03_growth.py             │  creature grows organically
│   ├── 04_multi_creature.py     │  two creatures, shared world
│   └── 05_ecosystem.py          │  full ecosystem w/ communication
│
│   ════════════════════════════════════════════════════
│   CORE LIBRARY — stable infrastructure
│   ════════════════════════════════════════════════════
│
├── yeppoh/                      THE PACKAGE
│   ├── __init__.py
│   │
│   ├── body/                    PHYSICS LAYER (Genesis)
│   │   ├── __init__.py          │  public API: CreatureBody
│   │   ├── creature.py          │  assembles entities into a creature
│   │   ├── materials.py         │  material catalog (flesh, bone, muscle)
│   │   ├── morphology.py        │  ★ body plan definitions (add new creatures here)
│   │   └── actuators.py         │  muscle actuation interface
│   │
│   ├── senses/                  SENSORY LAYER
│   │   ├── __init__.py          │  public API: SensorySystem
│   │   ├── echolocation.py      │  Lidar raycasts → distance array
│   │   ├── touch.py             │  contact force sensing
│   │   ├── vision.py            │  egocentric depth/RGB camera
│   │   ├── proprioception.py    │  body state (position, velocity, strain)
│   │   └── chemoreception.py    │  pheromone gradient sensing
│   │
│   ├── world/                   ENVIRONMENT SYSTEMS (torch, external to Genesis)
│   │   ├── __init__.py          │  public API: WorldState
│   │   ├── pheromones.py        │  3D diffusion grid on GPU
│   │   ├── energy.py            │  metabolic bookkeeping
│   │   ├── stimuli.py           │  light sources, temperature, obstacles
│   │   └── reaction_diffusion.py│  Gray-Scott Turing patterns
│   │
│   ├── brain/                   COGNITIVE ARCHITECTURE (policy networks)
│   │   ├── __init__.py          │  public API: CreatureBrain
│   │   ├── encoder.py           │  sensory obs → feature vector
│   │   ├── memory.py            │  GRU temporal memory
│   │   ├── communication.py     │  ★ inter-agent message channel
│   │   ├── drives.py            │  internal state (hunger, curiosity, fear)
│   │   ├── world_model.py       │  latent dynamics predictor (optional)
│   │   └── policy.py            │  action heads (motor, signal, sense)
│   │
│   ├── env/                     ENVIRONMENT INTERFACE
│   │   ├── __init__.py          │  public API: YeppohEnv
│   │   ├── scene.py             │  Genesis scene builder
│   │   ├── multi_agent.py       │  PettingZoo ParallelEnv wrapper
│   │   ├── agent_manager.py     │  cell cluster lifecycle (split/merge)
│   │   └── rewards.py           │  ★ reward functions (add new rewards here)
│   │
│   ├── training/                TRAINING HARNESS
│   │   ├── __init__.py          │  public API: run_experiment
│   │   ├── runner.py            │  experiment runner (config → results)
│   │   ├── algorithms.py        │  algorithm registry + TorchRL/BenchMARL setup
│   │   ├── callbacks.py         │  logging, checkpointing, visualization
│   │   └── curriculum.py        │  difficulty scheduling
│   │
│   └── export/                  OUTPUT PIPELINE
│       ├── __init__.py
│       ├── mesh.py              │  particles → trimesh → glTF/GLB
│       ├── video.py             │  Genesis recorder wrapper
│       └── gallery.py           │  WebXR scene generator (Three.js)
│
├── scripts/                     CLI ENTRY POINTS
│   ├── train.py                 │  python scripts/train.py --config configs/default.yaml
│   ├── visualize.py             │  python scripts/visualize.py checkpoint.pt
│   ├── export.py                │  python scripts/export.py --format glb
│   └── evaluate.py              │  python scripts/evaluate.py --episodes 10
│
└── tests/
    ├── test_body.py
    ├── test_senses.py
    ├── test_world.py
    ├── test_brain.py
    └── test_env.py
```

---

## Data Flow

How observations and actions flow through the system each step:

```
                          STEP t
                            │
    ┌───────────────────────┼───────────────────────┐
    │     GENESIS SCENE     │                       │
    │                       ▼                       │
    │              scene.step()                     │
    │         (physics advances 1 tick)             │
    │                       │                       │
    │    ┌──────────────────┼──────────────────┐    │
    │    │                  │                  │    │
    │    ▼                  ▼                  ▼    │
    │  body              senses             world   │
    │  positions         readings           fields  │
    └────┬──────────────────┬──────────────────┬────┘
         │                  │                  │
         ▼                  ▼                  ▼
    ┌─────────────────────────────────────────────┐
    │              PER-AGENT OBS ASSEMBLY          │
    │                                              │
    │  proprio(28) + echo(16) + chemo(9) + ...     │
    │  + light(4) + touch(6) + messages(32)        │
    │  + drives(4)                                 │
    │                         total: ~99 dims      │
    └──────────────────────┬───────────────────────┘
                           │
                           ▼
    ┌──────────────────────────────────────────────┐
    │              AGENT BRAIN                      │
    │                                               │
    │  obs ──► encoder ──► memory(GRU) ──► policy   │
    │              ▲           │              │      │
    │              │           ▼              │      │
    │          messages    drives             │      │
    │          from        update             │      │
    │          neighbors                      │      │
    │                                         │      │
    │                              ┌──────────┘      │
    │                              ▼                  │
    │                     ┌────────────────┐          │
    │                     │   ACTIONS      │          │
    │                     │                │          │
    │                     │ motor     (27) │          │
    │                     │ signal    (16) │          │
    │                     │ sensing    (4) │          │
    │                     │      total: 47 │          │
    │                     └───────┬────────┘          │
    └─────────────────────────────┼────────────────────┘
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              ┌──────────┐ ┌──────────┐ ┌───────────┐
              │  BODY    │ │  WORLD   │ │  SENSES   │
              │          │ │          │ │           │
              │ actuate  │ │ emit     │ │ cast rays │
              │ muscles  │ │ pheromone│ │ (if agent │
              │ emit     │ │ consume  │ │  chose to)│
              │ particles│ │ energy   │ │           │
              └──────────┘ └──────────┘ └───────────┘
                    │             │             │
                    └─────────────┼─────────────┘
                                  │
                                  ▼
                          STEP t+1 ...
```

---

## Multi-Agent Structure

```
    ┌─────────── One Creature ─────────────┐
    │                                       │
    │  ┌─────────┐         ┌─────────┐     │
    │  │ Agent 0 │◄──phys──►│ Agent 1 │     │
    │  │ "core"  │ ──msg──► │ "limb"  │     │
    │  │         │          │         │     │
    │  │ 120 MPM │          │ 80 MPM  │     │
    │  │particles│          │particles│     │
    │  └────┬────┘          └────┬────┘     │
    │       │   ◄──physics──►    │           │
    │  ┌────┴────┐          ┌────┴────┐     │
    │  │ Agent 2 │◄──phys──►│ Agent 3 │     │
    │  │ "limb"  │ ──msg──► │ "growth"│     │
    │  │         │          │ tip     │     │
    │  │ 80 MPM  │          │ 60 MPM  │     │
    │  │particles│          │particles│     │
    │  └─────────┘          └─────────┘     │
    │                                       │
    │  Shared: creature-level reward        │
    │  Individual: local survival reward    │
    │  Communication: learned 16-dim msgs   │
    └───────────────────────────────────────┘

    PettingZoo agents:
      "creature_0_agent_0"  (core)
      "creature_0_agent_1"  (limb)
      "creature_0_agent_2"  (limb)
      "creature_0_agent_3"  (growth tip)

    CTDE (MAPPO):
      Centralized critic sees ALL agents' obs
      Each actor sees ONLY its own obs + messages
```

### Agent Lifecycle

```
    Agent split (growth):
    ┌───────┐         ┌───────┐  ┌───────┐
    │  200  │  ──►    │  120  │  │  80   │
    │ parts │  split  │ parts │  │ parts │
    └───────┘         └───────┘  └───────┘
                       parent     child
                       (keeps     (inherits
                        policy)    policy +
                                   mutation)

    Agent merge (fusion):
    ┌───────┐  ┌───────┐         ┌───────┐
    │  60   │  │  40   │  ──►    │  100  │
    │ parts │  │ parts │  merge  │ parts │
    └───────┘  └───────┘         └───────┘
     agent A    agent B           agent A
                                  (A's policy
                                   survives)
```

---

## Config System

Configs compose via Hydra-style overrides. Each YAML controls one dimension:

```yaml
# configs/default.yaml — the base experiment
creature: blob           # which body plan
brain: reactive           # which cognitive architecture
reward: locomotion        # which reward function
training: mappo           # which RL algorithm
world:
  pheromones: true
  n_channels: 3
  energy: true
scene:
  n_envs: 256
  dt: 0.01
  max_steps: 1000
```

Run experiments by overriding:

```bash
# Basic blob learning to move
python scripts/train.py --config configs/default.yaml

# Coral with social brain and growth reward
python scripts/train.py \
  creature=coral \
  brain=social \
  reward=growth \
  training=mappo

# Quick test with fewer envs
python scripts/train.py scene.n_envs=16 training.timesteps=10000
```

---

## How to Add New Things

### New creature body plan

```python
# yeppoh/body/morphology.py — add a function:

def build_starfish(scene, cfg):
    """5-armed radial creature."""
    core = scene.add_entity(
        material=MATERIALS["muscle"],
        morph=gs.morphs.Sphere(radius=0.2),
    )
    arms = []
    for i in range(5):
        angle = i * 2 * math.pi / 5
        arm = scene.add_entity(
            material=MATERIALS["elastic"],
            morph=gs.morphs.Cylinder(radius=0.05, height=0.4, pos=(...)),
        )
        arm.set_particle_constraints(...)  # attach to core
        arms.append(arm)
    return CreatureBody(core=core, limbs=arms)
```

Then add `configs/creature/starfish.yaml`.

### New sense

```python
# yeppoh/senses/magnetoreception.py

class Magnetoreception(SenseModule):
    """Sense orientation relative to a global magnetic field."""
    
    def setup(self, scene, entity, cfg):
        self.field_direction = torch.tensor(cfg.field_direction)
    
    def read(self, entity_state) -> torch.Tensor:
        body_orientation = entity_state.quaternion
        relative = quaternion_apply(body_orientation, self.field_direction)
        return relative  # (3,) — how the field looks from the creature's frame
```

Register it in `yeppoh/senses/__init__.py`.

### New reward

```python
# yeppoh/env/rewards.py — add a function:

@reward_registry.register("curiosity")
def curiosity_reward(obs, action, next_obs, world_model):
    """Intrinsic reward from prediction error."""
    predicted = world_model.predict(obs, action)
    error = F.mse_loss(predicted, next_obs, reduction="none").mean(-1)
    return error.clamp(0, 1.0)  # surprise = reward
```

Then use it: `python scripts/train.py reward=curiosity`
