"""Visualize a trained sculpture agent.

Usage:
    python visualize.py checkpoints/sculpture_final
    python visualize.py checkpoints/sculpture_final --export-mesh
    python visualize.py checkpoints/sculpture_final --stochastic  # varied results
"""

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import imageio.v2 as imageio

from stable_baselines3 import PPO
from sculpture.env import SculptureEnv


def render_mesh(mesh, ax, title=None, color="teal", alpha=0.7):
    """Render a trimesh onto a matplotlib 3D axis."""
    ax.clear()
    vertices = mesh.vertices
    faces = mesh.faces

    polys = [[vertices[i] for i in face] for face in faces]
    collection = Poly3DCollection(polys, alpha=alpha)
    collection.set_facecolor(color)
    collection.set_edgecolor("black")
    collection.set_linewidth(0.1)
    ax.add_collection3d(collection)

    lim = np.max(np.abs(vertices)) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if title:
        ax.set_title(title)


def run_episode(model, env, deterministic=True):
    """Run one episode, return list of mesh snapshots and rewards."""
    obs, _ = env.reset()
    meshes = [env.get_mesh_snapshot()]
    rewards = []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        meshes.append(env.get_mesh_snapshot())
        rewards.append(reward)

    return meshes, rewards


def create_animation(meshes, output_path, fps=15, step_interval=5):
    """Create GIF of the sculpture growing over time."""
    fig = plt.figure(figsize=(8, 8), dpi=80)
    ax = fig.add_subplot(111, projection="3d")
    frames = []

    indices = list(range(0, len(meshes), step_interval))
    if indices[-1] != len(meshes) - 1:
        indices.append(len(meshes) - 1)

    for i, idx in enumerate(indices):
        render_mesh(meshes[idx], ax, title=f"Step {idx}/{len(meshes)-1}")
        ax.view_init(elev=20, azim=idx * 1.5)

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(img)
        print(f"  Frame {i+1}/{len(indices)}", end="\r")

    plt.close(fig)
    imageio.mimsave(str(output_path), frames, fps=fps, loop=0)
    print(f"\nAnimation saved → {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize trained sculpture")
    parser.add_argument("model_path", help="Path to trained model (.zip)")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--step-interval", type=int, default=3)
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--export-mesh", action="store_true",
                        help="Export final mesh as .glb")
    parser.add_argument("--stochastic", action="store_true",
                        help="Use stochastic policy for varied results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env = SculptureEnv(subdivisions=2, n_anchors=16, max_steps=200)
    model = PPO.load(args.model_path)

    print("Running episode...")
    meshes, rewards = run_episode(model, env, deterministic=not args.stochastic)

    print(f"Episode: {len(meshes)} steps, total reward: {sum(rewards):.2f}")
    print(f"Surface area: {meshes[-1].area / meshes[0].area:.2f}x initial")

    # Save key-frame snapshots
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    for frac in [0, 0.25, 0.5, 0.75, 1.0]:
        idx = min(int(frac * (len(meshes) - 1)), len(meshes) - 1)
        render_mesh(meshes[idx], ax, title=f"Step {idx}")
        fig.savefig(output_dir / f"snapshot_{idx:04d}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Snapshots → {output_dir}/")

    if not args.no_gif:
        print("Creating animation...")
        create_animation(meshes, output_dir / "sculpture.gif",
                         fps=args.fps, step_interval=args.step_interval)

    if args.export_mesh:
        out = output_dir / "sculpture_final.glb"
        meshes[-1].export(str(out))
        print(f"Mesh exported → {out}")


if __name__ == "__main__":
    main()
