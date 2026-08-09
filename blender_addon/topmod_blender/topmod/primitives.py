"""
Primitive mesh generators.

Each function returns a DLFLMesh that is a valid, closed, orientable
2-manifold.  The meshes are built by directly wiring the DLFL half-edge
structure rather than going through the high-level operators (faster and
avoids bootstrapping issues).

Public API
----------
make_cube()         -> DLFLMesh    (6 quad faces,  8V 12E  6F)
make_tetrahedron()  -> DLFLMesh    (4 tri  faces,  4V  6E  4F)
make_icosahedron()  -> DLFLMesh    (20 tri faces, 12V 30E 20F)
"""

from __future__ import annotations
import math
from typing import Dict, List, Tuple

from .dlfl import DLFLMesh, Vertex, HalfEdge, Face, Edge


# ── low-level builder helpers ─────────────────────────────────────────────────

def _build_mesh(positions: List[Tuple[float, float, float]],
                face_indices: List[List[int]]) -> DLFLMesh:
    """
    Build a DLFLMesh from a vertex position list and face-vertex index lists.

    Each face in face_indices is a list of vertex indices (CCW winding).
    All positions must be referenced by at least one face.
    """
    mesh = DLFLMesh()

    # Create vertices
    verts: List[Vertex] = []
    for x, y, z in positions:
        verts.append(mesh._new_vertex(x, y, z))

    # For each face, create the ring of half-edges
    # We also need to pair up twins.
    # Key: frozenset({v_i, v_j}) -> list of half-edges on that undirected edge
    edge_map: Dict[Tuple[int, int], HalfEdge] = {}
    # edge_map[(i, j)] = half-edge from i to j

    for fidx, vindex_list in enumerate(face_indices):
        face = mesh._new_face()
        n = len(vindex_list)

        # Create half-edges for this face
        hes: List[HalfEdge] = []
        for k in range(n):
            he = mesh._new_halfedge()
            he.origin = verts[vindex_list[k]]
            he.face   = face
            hes.append(he)
            edge_map[(vindex_list[k], vindex_list[(k + 1) % n])] = he

        # Wire next / prev within the face
        for k in range(n):
            hes[k].next = hes[(k + 1) % n]
            hes[k].prev = hes[(k - 1) % n]

        face.he = hes[0]

        # Set vertex.he (first time seen)
        for k in range(n):
            v = verts[vindex_list[k]]
            if v.he is None:
                v.he = hes[k]

    # Pair twins and create Edge objects
    paired: set[Tuple[int, int]] = set()
    for (vi, vj), he_ij in edge_map.items():
        if (vi, vj) in paired:
            continue
        he_ji = edge_map.get((vj, vi))
        if he_ji is None:
            raise ValueError(
                f"No twin for half-edge {vi}→{vj}: "
                "mesh is not closed or has boundary."
            )
        mesh._new_edge(he_ij, he_ji)
        paired.add((vi, vj))
        paired.add((vj, vi))

    return mesh


# ── cube ──────────────────────────────────────────────────────────────────────

def make_cube(size: float = 1.0) -> DLFLMesh:
    """
    Axis-aligned cube centred at the origin.

        6──7
       /|  /|
      4─+─5 |
      | 2─|─3
      |/  |/
      0───1

    Vertex layout (±h, ±h, ±h)  h = size / 2.
    6 quad faces, CCW winding viewed from outside.
    V=8, E=12, F=6, χ=2, genus=0.
    """
    h = size / 2.0
    positions = [
        (-h, -h, -h),  # 0
        ( h, -h, -h),  # 1
        (-h,  h, -h),  # 2
        ( h,  h, -h),  # 3
        (-h, -h,  h),  # 4
        ( h, -h,  h),  # 5
        (-h,  h,  h),  # 6
        ( h,  h,  h),  # 7
    ]
    faces = [
        [0, 2, 3, 1],  # bottom  (-z)
        [4, 5, 7, 6],  # top     (+z)
        [0, 1, 5, 4],  # front   (-y)
        [2, 6, 7, 3],  # back    (+y)
        [0, 4, 6, 2],  # left    (-x)
        [1, 3, 7, 5],  # right   (+x)
    ]
    return _build_mesh(positions, faces)


# ── tetrahedron ───────────────────────────────────────────────────────────────

def make_tetrahedron(size: float = 1.0) -> DLFLMesh:
    """
    Regular tetrahedron.
    V=4, E=6, F=4, χ=2, genus=0.
    """
    s = size
    positions = [
        ( s,  s,  s),   # 0
        ( s, -s, -s),   # 1
        (-s,  s, -s),   # 2
        (-s, -s,  s),   # 3
    ]
    faces = [
        [0, 1, 3],
        [0, 3, 2],
        [0, 2, 1],
        [1, 2, 3],
    ]
    return _build_mesh(positions, faces)


# ── icosahedron ───────────────────────────────────────────────────────────────

def make_icosahedron(radius: float = 1.0) -> DLFLMesh:
    """
    Regular icosahedron.
    V=12, E=30, F=20, χ=2, genus=0.
    """
    phi = (1.0 + math.sqrt(5.0)) / 2.0

    # 12 vertices: three mutually perpendicular golden rectangles
    raw = [
        (-1,  phi, 0), ( 1,  phi, 0), (-1, -phi, 0), ( 1, -phi, 0),
        ( 0, -1,  phi), ( 0,  1,  phi), ( 0, -1, -phi), ( 0,  1, -phi),
        ( phi, 0, -1), ( phi, 0,  1), (-phi, 0, -1), (-phi, 0,  1),
    ]
    norm = math.sqrt(1 + phi**2)
    positions = [(x * radius / norm, y * radius / norm, z * radius / norm)
                 for x, y, z in raw]

    # 20 triangular faces (CCW from outside)
    faces = [
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9],  [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4],  [3, 4, 2],  [3, 2, 6],  [3, 6, 8],  [3, 8, 9],
        [4, 9, 5],  [2, 4, 11], [6, 2, 10], [8, 6, 7],  [9, 8, 1],
    ]
    return _build_mesh(positions, faces)


# ── octahedron (bonus) ────────────────────────────────────────────────────────

def make_octahedron(size: float = 1.0) -> DLFLMesh:
    """
    Regular octahedron.
    V=6, E=12, F=8, χ=2, genus=0.
    """
    s = size
    positions = [
        ( s, 0, 0), (-s, 0, 0),
        ( 0, s, 0), ( 0,-s, 0),
        ( 0, 0, s), ( 0, 0,-s),
    ]
    faces = [
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
    ]
    return _build_mesh(positions, faces)
