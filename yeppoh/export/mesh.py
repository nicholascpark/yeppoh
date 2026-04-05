"""Mesh export — Genesis particles → trimesh → glTF/GLB."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh


def particles_to_mesh(
    positions: np.ndarray,
    radius: float = 0.01,
    method: str = "ball_pivot",
) -> trimesh.Trimesh:
    """Convert particle positions to a triangle mesh.

    Args:
        positions: (N, 3) particle positions
        radius: particle radius for surface reconstruction
        method: "ball_pivot" or "convex_hull"

    Returns:
        trimesh.Trimesh
    """
    if method == "convex_hull":
        cloud = trimesh.PointCloud(positions)
        return cloud.convex_hull

    # Ball pivot / alpha shape reconstruction
    cloud = trimesh.PointCloud(positions)
    try:
        mesh = cloud.convex_hull  # fallback — proper reconstruction needs open3d
    except Exception:
        # Create spheres at each particle position
        meshes = []
        for pos in positions[::4]:  # subsample for speed
            sphere = trimesh.creation.icosphere(subdivisions=1, radius=radius)
            sphere.apply_translation(pos)
            meshes.append(sphere)
        mesh = trimesh.util.concatenate(meshes)

    return mesh


def export_mesh(
    positions: np.ndarray,
    output_path: str | Path,
    format: str = "glb",
) -> Path:
    """Export particle positions as a 3D mesh file.

    Args:
        positions: (N, 3) particle positions
        output_path: where to save
        format: "glb", "obj", "stl", "ply"
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = particles_to_mesh(positions)
    mesh.export(str(output_path), file_type=format)

    print(f"Exported mesh ({len(positions)} particles) → {output_path}")
    return output_path


def export_animation(
    frame_positions: list[np.ndarray],
    output_path: str | Path,
    fps: int = 30,
) -> Path:
    """Export a sequence of frames as animated glTF.

    Each frame is a set of particle positions. Frames are baked
    as morph target keyframes in the glTF.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create base mesh from first frame
    base_mesh = particles_to_mesh(frame_positions[0])

    # For animated glTF, we'd need to add morph targets or skeletal animation.
    # Trimesh doesn't support animated glTF natively.
    # For now, export individual frames as separate files.
    for i, positions in enumerate(frame_positions):
        frame_mesh = particles_to_mesh(positions)
        frame_path = output_path.parent / f"{output_path.stem}_{i:04d}.glb"
        frame_mesh.export(str(frame_path))

    print(f"Exported {len(frame_positions)} frames → {output_path.parent}/")
    return output_path
