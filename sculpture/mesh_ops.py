"""Mesh operations for sculpture deformation."""

import numpy as np
from scipy.spatial.distance import cdist
import trimesh


def create_base_mesh(subdivisions: int = 2) -> trimesh.Trimesh:
    """Create an icosphere base mesh.

    subdivisions=2 gives 162 vertices — enough detail to look organic,
    small enough to keep training fast.
    """
    return trimesh.creation.icosphere(subdivisions=subdivisions, radius=1.0)


def farthest_point_sampling(vertices: np.ndarray, k: int) -> np.ndarray:
    """Select k well-distributed vertices via farthest point sampling.

    Ensures anchor points are spread evenly across the mesh surface,
    giving each anchor a distinct region of influence.
    """
    n = len(vertices)
    selected = [0]
    min_distances = np.full(n, np.inf)

    for _ in range(k - 1):
        last = selected[-1]
        dists = np.linalg.norm(vertices - vertices[last], axis=1)
        min_distances = np.minimum(min_distances, dists)
        next_idx = np.argmax(min_distances)
        selected.append(int(next_idx))

    return np.array(selected)


def compute_rbf_weights(
    vertices: np.ndarray,
    anchor_positions: np.ndarray,
    sigma: float = 0.5,
) -> np.ndarray:
    """Compute RBF interpolation weights.

    Each vertex's displacement is a weighted average of anchor displacements,
    weighted by Gaussian proximity. This produces smooth, organic deformations.

    Returns (n_vertices, n_anchors) matrix where each row sums to 1.
    """
    dists = cdist(vertices, anchor_positions)
    weights = np.exp(-dists**2 / (2 * sigma**2))
    row_sums = np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    weights /= row_sums
    return weights.astype(np.float32)


def apply_deformation(
    mesh: trimesh.Trimesh,
    anchor_deltas: np.ndarray,
    rbf_weights: np.ndarray,
) -> None:
    """Apply RBF-interpolated deformation from anchor displacements (in-place)."""
    vertex_deltas = rbf_weights @ anchor_deltas
    mesh.vertices += vertex_deltas


def laplacian_smooth(
    mesh: trimesh.Trimesh,
    iterations: int = 1,
    factor: float = 0.3,
) -> None:
    """Apply Laplacian smoothing (in-place).

    Pulls each vertex toward the average of its neighbors, preventing
    spiky artifacts and keeping the surface organic.
    """
    edges = mesh.edges_unique
    n = len(mesh.vertices)

    # Build neighbor lists once
    neighbors: list[list[int]] = [[] for _ in range(n)]
    for a, b in edges:
        neighbors[a].append(b)
        neighbors[b].append(a)

    vertices = mesh.vertices.copy()
    for _ in range(iterations):
        new_verts = vertices.copy()
        for i, nbrs in enumerate(neighbors):
            if nbrs:
                avg = vertices[nbrs].mean(axis=0)
                new_verts[i] += factor * (avg - vertices[i])
        vertices = new_verts

    mesh.vertices = vertices
