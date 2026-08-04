"""
Topology Builder — Phase 2 of the GenesisTopmod pipeline.

Given a target topology spec (genus, boundary count, components), constructs
a DLFLMesh with the correct topology, then returns (vertices, faces) as GPU
tensors ready for differentiable geometry optimization.

Strategy
--------
1. Start from make_icosahedron() (genus-0, closed surface).
2. For each handle needed: find two compatible (non-adjacent) faces and call
   add_handle(mesh, f1, f2) to increase genus by 1.
3. Apply Catmull-Clark subdivision `subdivisions` times for resolution.
4. Triangulate via to_triangle_arrays() (fan triangulation of quads).
5. Return as float32 vertex positions [V, 3] + int32 face indices [F, 3].

Public API
----------
build_topology(genus, boundaries, subdivisions, scale, device) -> (verts, faces)
"""

from __future__ import annotations
import sys, os
import math
from typing import Set, Tuple, Optional

import numpy as np
import torch

# Ensure the repo root is in path when running standalone
_HERE = os.path.dirname(__file__)
_ROOT = os.path.join(_HERE, "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from topmod.primitives import make_icosahedron
from topmod.high_level_ops import add_handle
from topmod.subdivision import catmull_clark
from topmod.io import to_triangle_arrays
from topmod.dlfl import DLFLMesh, Face
from topmod.validate import is_manifold


# ── face-pair finder ─────────────────────────────────────────────────────────

def _find_compatible_face_pair(
    mesh: DLFLMesh,
    exclude_vids: Set[int],
    min_degree: int = 3,
) -> Tuple[Face, Face]:
    """
    Find two faces in *mesh* that:
    1. Each has degree >= min_degree.
    2. No vertex is in *exclude_vids*.
    3. They share no vertices with each other.

    Returns (f1, f2) ready to be passed to add_handle.

    Raises ValueError if no such pair exists.
    """
    candidates = []
    for f in mesh.iter_faces():
        if f.degree() < min_degree:
            continue
        vids = {v.id for v in f.vertices()}
        if vids & exclude_vids:
            continue
        candidates.append((f, vids))

    n = len(candidates)
    for i in range(n):
        f1, vids1 = candidates[i]
        for j in range(i + 1, n):
            f2, vids2 = candidates[j]
            if not (vids1 & vids2):   # no shared vertices
                return f1, f2

    raise ValueError(
        f"Cannot find two non-adjacent faces (candidates={n}, "
        f"excluded_vids={len(exclude_vids)}). "
        "Try subdividing the base mesh first."
    )


# ── topology builder ──────────────────────────────────────────────────────────

def build_topology(
    genus:        int   = 0,
    boundaries:   int   = 0,
    subdivisions: int   = 2,
    scale:        float = 1.0,
    device:       str   = "cuda",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a mesh with the target topology and return GPU tensors.

    Parameters
    ----------
    genus : int
        Number of topological handles (holes through the mesh).
        genus=0 → sphere-like;  genus=1 → torus-like.
    boundaries : int
        Number of boundary loops (open holes in the surface).  Currently
        implemented by removing faces after subdivision.
    subdivisions : int
        Number of Catmull-Clark subdivision rounds.  More = smoother and
        denser.  2 rounds ≈ 386 vertices for genus-0 icosahedron base.
    scale : float
        Uniform scale applied to all vertex positions.
    device : str
        PyTorch device string ("cuda" or "cpu").

    Returns
    -------
    verts : [V, 3] float32 tensor
    faces : [F, 3] int32 tensor
    """
    # ── 1. Base mesh ──────────────────────────────────────────────────
    mesh = make_icosahedron(radius=1.0)

    # ── 2. Add topological handles ────────────────────────────────────
    excluded_vids: Set[int] = set()

    for g in range(genus):
        if not is_manifold(mesh):
            raise RuntimeError(f"Mesh became non-manifold before adding handle {g + 1}")

        f1, f2 = _find_compatible_face_pair(mesh, excluded_vids)

        # Record vertex IDs before add_handle consumes these faces
        used = {v.id for v in f1.vertices()} | {v.id for v in f2.vertices()}

        add_handle(mesh, f1, f2)

        excluded_vids |= used   # don't touch these vertices for future handles

        if not is_manifold(mesh):
            raise RuntimeError(f"Mesh became non-manifold after adding handle {g + 1}")

    # ── 3. Catmull-Clark subdivision ──────────────────────────────────
    for _ in range(subdivisions):
        mesh = catmull_clark(mesh)

    # ── 4. Open boundaries (delete faces) ─────────────────────────────
    if boundaries > 0:
        faces_list = list(mesh.iter_faces())
        # Evenly space removed faces around the mesh
        step = max(1, len(faces_list) // (boundaries * 2))
        from topmod.operators import delete_edge
        removed = 0
        for i in range(0, len(faces_list), step):
            if removed >= boundaries:
                break
            f = faces_list[i]
            if f.id not in mesh.faces:
                continue
            # Open a hole: remove one edge of the face to merge with adjacent face,
            # then delete the merged region — simplified: just remove the face by
            # deleting its first edge (opens the mesh).
            # For a proper hole, we'd use a dedicated "delete_face" operator.
            # Here we do a minimal implementation: delete one edge adjacent to this face.
            he = f.he
            if he and he.edge and he.edge.id in mesh.edges:
                delete_edge(mesh, he.edge)
                removed += 1

    # ── 5. Convert to arrays ──────────────────────────────────────────
    positions, triangles = to_triangle_arrays(mesh)

    if len(triangles) == 0:
        raise ValueError("Topology builder produced a mesh with no triangles.")

    # Scale
    verts_np = np.array(positions, dtype=np.float32) * scale
    faces_np = np.array(triangles, dtype=np.int32)

    # Normalize so mesh centre is at origin (already is for icosahedron;
    # handle insertion may shift it slightly)
    centroid = verts_np.mean(axis=0, keepdims=True)
    verts_np -= centroid

    verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.int32,   device=device)

    return verts, faces


# ── genus verification helper ─────────────────────────────────────────────────

def verify_genus(verts: torch.Tensor, faces: torch.Tensor) -> int:
    """
    Compute the genus of the mesh given by (verts, faces) tensors.

    Uses Euler characteristic: χ = V - E + F = 2 - 2g (for closed surface).
    """
    V = verts.shape[0]
    F = faces.shape[0]

    # Count unique edges from face index data
    faces_np = faces.cpu().numpy()
    edges = set()
    for tri in faces_np:
        for k in range(3):
            e = tuple(sorted([int(tri[k]), int(tri[(k + 1) % 3])]))
            edges.add(e)
    E = len(edges)

    chi = V - E + F
    # For genus-g closed surface with 1 component: chi = 2 - 2g
    g = (2 - chi) // 2
    return int(g)
