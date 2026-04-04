# yeppoh

RL-driven "living sculptures" — 3D meshes deformed and grown by reinforcement learning agents, viewable in VR on Meta Quest and Apple Vision Pro.

## Recommended Stack

| Layer | Choice | Why |
|---|---|---|
| **Simulation** | Custom vertex-level + RBF interpolation | No physics engine overhead. Direct control over every vertex. For art, we want creative control, not physical accuracy |
| **RL** | Stable-Baselines3 (PPO) | Most mature, best docs, handles continuous 48-dim action space well. Easy to debug |
| **Reward** | Geometric heuristics → CLIP hybrid | Phase 1: fast math (surface area, curvature, radial variance). Phase 2: CLIP for subjective aesthetics |
| **Export** | Animated glTF/GLB via trimesh | Universal 3D format, works everywhere |
| **VR** | Three.js + WebXR | Fastest path to headset. Runs in Quest browser, no app store. Can iterate in minutes |

**Why not the alternatives?**
- *Taichi*: Powerful but adds complexity we don't need yet. Can layer it in for Phase 5 reactive physics.
- *Brax*: JAX-based and fast, but rigid-body focused — wrong primitive for organic mesh deformation.
- *PyBullet*: Too slow for the training loop; soft body support is clunky.
- *Unity*: Better for polished final product, but slower iteration. Save for Phase 6.

## Quick Start

```bash
cd yeppoh
pip install -r requirements.txt

# Train (~1-2 hours GPU, check progress with TensorBoard)
python train.py --timesteps 500000

# Monitor
tensorboard --logdir logs/

# Visualize
python visualize.py checkpoints/sculpture_final --export-mesh

# Output: output/sculpture.gif + output/sculpture_final.glb
```

## How It Works

An RL agent (PPO) controls **16 anchor points** on an icosphere (162 vertices). Each step:

1. Agent outputs small 3D displacements for each anchor (action space: 48 dims)
2. Displacements are smoothly interpolated to all vertices via **RBF kernels**
3. **Laplacian smoothing** prevents spikes, keeps surface organic
4. Reward = surface area growth + radial variance + smoothness - degeneracy penalty
5. Over 200 steps, the sphere grows into something alien and alive

## Pipeline Cheat Sheet

```
TRAIN                    EXPORT                 VIEW
─────                    ──────                 ────

┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  PPO Agent   │──▶│  Bake Mesh   │──▶│  Three.js    │──▶│  Quest /     │
│  (SB3)       │   │  Frames      │   │  WebXR Scene │   │  Vision Pro  │
│              │   │              │   │              │   │              │
│  Python      │   │  .glb file   │   │  HTML/JS     │   │  Browser     │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
      │                                      │
      ▼                                      ▼
  TensorBoard                          localhost:8080
  (reward curves)                      (dev server)

File formats at each stage:
  .zip (SB3 checkpoint) → .glb (mesh) → .html (scene) → WebXR (headset)
```

## MVP Definition

**Simplest working version:** A trained PPO agent that takes a sphere and deforms it into something that looks organic and intentional (not random noise). Viewable as an animated GIF and exportable as .glb.

- **Input:** Icosphere (162 vertices)
- **Output:** Deformed mesh (.glb) + growth animation (.gif)
- **Intermediate:** SB3 checkpoint (.zip), TensorBoard logs
- **Observation space:** 492 dims (162×3 normalized vertex positions + 6 mesh stats)
- **Action space:** 48 dims (16 anchors × 3 axes, scaled to ±0.02 per step)
- **GPU training time:** ~1-2 hours for 500K steps on consumer NVIDIA GPU

## Roadmap

### Phase 1 — Proof of Concept ✅ (this repo)
Train one agent that visibly deforms a mesh in an interesting way. Validate the env, reward, and training loop.

### Phase 2 — Aesthetic Reward
- Add CLIP-based scoring ("looks like living coral", "organic alien structure")
- Hybrid reward: 70% geometric + 30% CLIP
- Experiment with different text prompts as style control

### Phase 3 — Export Pipeline
- Bake training trajectory into animated glTF with keyframes
- Blender script for offline beauty renders (subsurface scattering, caustics)
- Multiple camera angles, turntable animation

### Phase 4 — VR Gallery
- Three.js + WebXR scene with pedestal lighting
- Load .glb sculptures into a gallery room
- Basic interaction: walk around, look closely
- Test on Meta Quest browser

### Phase 5 — Reactive Sculptures
- Ship the trained policy (ONNX) to the browser
- Run inference in real-time via ONNX.js
- Sculptures respond to viewer proximity (grow toward you)
- Hand tracking input → deformation influence

### Phase 6 — Polish
- Multi-sculpture gallery with different trained styles
- Spatial audio (generative ambient tied to deformation rate)
- Volumetric lighting, particle effects
- Apple Vision Pro support via visionOS WebXR

## Project Structure

```
yeppoh/
├── sculpture/
│   ├── __init__.py
│   ├── env.py          # Gymnasium environment
│   ├── mesh_ops.py     # Mesh deformation utilities
│   └── rewards.py      # Reward functions
├── train.py            # Training script (SB3 PPO)
├── visualize.py        # Visualization + GIF + mesh export
├── requirements.txt
└── README.md
```
