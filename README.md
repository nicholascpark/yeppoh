# yeppoh

Multi-agent RL creatures with emergent behavior — soft-body organisms that sense, communicate, grow, and develop in a physics simulation. Viewable in VR.

## Stack

| Layer | Tool | Role |
|---|---|---|
| Physics | Genesis (MPM + FEM + SPH) | Soft body simulation, muscle actuation, fluid coupling |
| Multi-Agent Env | PettingZoo | Standard MARL environment API |
| RL Training | TorchRL / BenchMARL | MAPPO, QMIX, IPPO, MADDPG |
| Cognitive Arch | PyTorch (custom) | Encoder → GRU memory → communication → drives → policy |
| World Systems | PyTorch (custom) | Pheromone diffusion, metabolism, reaction-diffusion |
| VR Export | Three.js + WebXR | Browser-based VR gallery |

## Quick Start

```bash
pip install -r requirements.txt

# Experiment 01: basic blob learns to pulse
python experiments/01_basic_blob.py

# Experiment 02: blob learns to crawl
python experiments/02_locomotion.py

# Full config-driven training
python scripts/train.py

# Override any config key via CLI
python scripts/train.py creature.plan=coral training.algorithm=ippo

# Visualize trained creature
python scripts/visualize.py runs/default/final.pt
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full visual code layout.

```
configs/           ← experiment knobs (creature, brain, reward, algorithm)
experiments/       ← runnable experiment scripts
yeppoh/
  body/            ← Genesis physics (creatures, materials, muscles)
  senses/          ← perception (echolocation, touch, vision, chemoreception)
  world/           ← environment systems (pheromones, energy, reaction-diffusion)
  brain/           ← cognitive architecture (encoder, memory, communication, drives)
  env/             ← PettingZoo multi-agent wrapper
  training/        ← experiment runner, algorithm registry, callbacks
  export/          ← mesh → glTF, video recording, WebXR gallery
scripts/           ← CLI entry points (train, visualize, export)
```

## Key Design Decisions

**Each creature is a team of agents.** A blob with 4 agents means 4 cell clusters
coordinating through shared physics + learned communication. Roles (muscle, skeleton,
sensor) emerge from multi-agent optimization pressure — they're not programmed.

**Sensing is an action.** Echolocation costs energy. The agent decides whether to ping.
This means attention itself is learned.

**Metabolism forces tradeoffs.** An agent can't grow, move, AND signal at full power.
Specialization emerges from energy pressure.

**Communication is differentiable.** Agents broadcast learned vectors. Over training,
they develop functional protocols — emergent language.
