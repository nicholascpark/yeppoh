"""Reward functions for sculpture RL environment.

Phase 1 uses pure geometric heuristics — fast to compute, no external models.
Phase 2 will layer CLIP-based aesthetic scoring on top.
"""

import numpy as np
import trimesh


def compute_reward(
    mesh: trimesh.Trimesh,
    initial_mesh: trimesh.Trimesh,
    prev_vertices: np.ndarray | None,
    step: int,
    max_steps: int,
) -> tuple[float, dict]:
    """Compute composite reward for current mesh state.

    Reward components:
    - surface_area: encourages topological complexity (coral-like growth)
    - volume: encourages expansion, penalizes collapse
    - radial_variance: encourages asymmetric, interesting shapes
    - smoothness: penalizes spiky/degenerate geometry
    - movement: encourages continued exploration, prevents freezing
    """
    components: dict[str, float] = {}

    # --- Surface area growth ---
    sa_ratio = mesh.area / initial_mesh.area
    components["surface_area"] = float(np.clip(sa_ratio - 1.0, 0.0, 3.0) * 0.4)

    # --- Volume growth ---
    try:
        vol_ratio = abs(mesh.volume) / abs(initial_mesh.volume)
        components["volume"] = float(np.clip(vol_ratio - 1.0, 0.0, 3.0) * 0.2)
    except Exception:
        components["volume"] = 0.0

    # --- Radial variance (shape interestingness) ---
    centroid = mesh.vertices.mean(axis=0)
    radii = np.linalg.norm(mesh.vertices - centroid, axis=1)
    radial_cv = float(radii.std() / (radii.mean() + 1e-8))
    components["radial_variance"] = float(np.clip(radial_cv, 0.0, 1.5) * 0.5)

    # --- Smoothness penalty ---
    normal_var = float(np.var(mesh.face_normals, axis=0).sum())
    # Sphere has ~0.67; penalize above 1.5 (spiky territory)
    components["smoothness"] = -0.3 * max(0.0, normal_var - 1.5)

    # --- Movement reward ---
    if prev_vertices is not None:
        movement = float(np.mean(np.linalg.norm(
            mesh.vertices - prev_vertices, axis=1
        )))
        components["movement"] = float(np.clip(movement * 10.0, 0.0, 0.3))
    else:
        components["movement"] = 0.0

    # --- Degenerate face penalty ---
    min_area = float(mesh.area_faces.min())
    components["degenerate"] = -1.0 if min_area < 1e-6 else 0.0

    total = sum(components.values())
    return total, components
