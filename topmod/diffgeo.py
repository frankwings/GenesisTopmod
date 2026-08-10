"""
topmod/diffgeo.py — differentiable (PyTorch) geometry for TopMod operators.

Topology stays discrete (which vertices/edges/faces exist and how they are
connected is decided by the operator and carries no gradient); geometry
(where the vertices are placed) becomes a differentiable torch function of
the input vertex positions.

Three mechanisms
----------------
1. **Linear operators** (Catmull-Clark, Doo-Sabin, Loop, and all other
   schemes whose new positions are sparse linear combinations of the input
   positions): *symbolic trace*.  The existing float implementation is
   re-run with every coordinate replaced by a symbolic linear-combination
   object (`_Lin`); the output "coordinates" literally are the rows of the
   sparse weight matrix ``W``, so ``new_verts = W @ old_verts``.  The float
   implementation remains the single source of truth — no geometry rule is
   duplicated here.  Any nonlinear use of coordinates (products, sqrt,
   comparisons) raises immediately during tracing, so a wrongly classified
   operator cannot silently produce bad gradients.

2. **Nonlinear operators** (STAR, FRAC, DOME, CRUST, EXTRUDE_FACE,
   STELLATE, SUBDIVIDE_EDGE, SUBDIVIDE_FACE): dedicated torch
   implementations, verified against the float path by the test suite.
   Continuous parameters (offset, dist, thickness, ...) may be passed as
   torch scalar tensors and receive gradients.

3. **Topology-only operators** (IE, DE, HDL): the position map is the
   identity — nothing to differentiate.

Public API
----------
mesh_to_arrays(mesh)                     -> (positions, faces)
trace_op(name, positions, faces, **kw)   -> DiffOp
DiffOp.apply(verts [V_in,3])             -> verts [V_out,3]  (differentiable)
DiffSequence(base, ...).append(name, **kw)
DiffSequence.forward(verts0=None)        -> final verts tensor
DiffSequence.triangles()                 -> int64 tensor [T,3] (fan triangulated)
LINEAR_OPS / NONLINEAR_OPS               -> supported opcode tuples
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch

from .dlfl import DLFLMesh
from .primitives import _build_mesh, make_cube, make_tetrahedron, make_icosahedron
from .subdivision import catmull_clark
from .remeshing import (
    dual, doo_sabin, simplest_subdivide, vertex_cutting,
    loop_subdivide, sqrt3_subdivide,
    honeycomb_subdivide, corner_cutting,
    loop_style_subdivide,
    pentagonal_subdivide, pentagonal2_subdivide,
    dual1264_subdivide, root4_subdivide,
    checkerboard_remesh, ds_bc_new_subdivide,
    star_subdivide, fractal_subdivide, dome_subdivide,
    create_crust,
    _DOME_HEIGHTS, _DOME_SCALES,
)
from .high_level_ops import (
    stellate_all, stellate, extrude_face, subdivide_edge, subdivide_face,
    add_handle,
)


# ─────────────────────────────────────────────────────────────────────────────
# Symbolic linear-combination scalar for tracing
# ─────────────────────────────────────────────────────────────────────────────

class _NonLinearTrace(TypeError):
    """Raised when a traced operator uses coordinates non-linearly."""


class _Lin:
    """
    A symbolic scalar: sum_i c[i] * x_i (+ const), where x_i is the
    corresponding coordinate of input vertex i.

    Supports exactly the arithmetic a *linear* operator may perform on
    coordinates: +, -, unary -, multiplication/division by numbers.
    Anything else (Lin*Lin, sqrt, comparisons, ...) raises, which is the
    tracer's built-in proof that the operator really is linear.
    """

    __slots__ = ("c", "k")

    def __init__(self, c: Dict[int, float], k: float = 0.0):
        self.c = c
        self.k = k

    # -- addition ---------------------------------------------------------
    def __add__(self, other):
        if isinstance(other, _Lin):
            c = dict(self.c)
            for i, w in other.c.items():
                c[i] = c.get(i, 0.0) + w
            return _Lin(c, self.k + other.k)
        if isinstance(other, (int, float)):
            return _Lin(dict(self.c), self.k + other)
        return NotImplemented

    __radd__ = __add__

    # -- subtraction ------------------------------------------------------
    def __sub__(self, other):
        return self.__add__(-other if isinstance(other, _Lin) else -other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __neg__(self):
        return _Lin({i: -w for i, w in self.c.items()}, -self.k)

    def __pos__(self):
        return self

    # -- scaling ----------------------------------------------------------
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return _Lin({i: w * other for i, w in self.c.items()},
                        self.k * other)
        raise _NonLinearTrace(
            "coordinate * coordinate during trace — operator is not linear")

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return _Lin({i: w / other for i, w in self.c.items()},
                        self.k / other)
        raise _NonLinearTrace(
            "division by a coordinate during trace — operator is not linear")

    # -- everything else is a linearity violation -------------------------
    def _nonlinear(self, *_a, **_k):
        raise _NonLinearTrace("non-linear coordinate use during trace")

    __pow__ = __rtruediv__ = __abs__ = _nonlinear
    __lt__ = __le__ = __gt__ = __ge__ = _nonlinear

    def __float__(self):
        raise _NonLinearTrace(
            "float(coordinate) during trace (math.sqrt etc.) — "
            "operator is not linear")

    def __repr__(self):
        return f"_Lin({self.c}, k={self.k})"


# ─────────────────────────────────────────────────────────────────────────────
# Mesh <-> arrays
# ─────────────────────────────────────────────────────────────────────────────

def mesh_to_arrays(mesh: DLFLMesh) -> Tuple[List[Tuple[float, float, float]],
                                            List[List[int]]]:
    """
    Export a DLFLMesh as (positions, polygon face index lists) using the
    stable ``mesh.vertices.values()`` / ``mesh.faces.values()`` iteration
    order (the same ordering io.to_triangle_arrays uses).
    """
    vid_to_idx: Dict[int, int] = {}
    positions: List[Tuple[float, float, float]] = []
    for i, v in enumerate(mesh.vertices.values()):
        vid_to_idx[v.id] = i
        positions.append((v.x, v.y, v.z))
    faces: List[List[int]] = []
    for f in mesh.faces.values():
        faces.append([vid_to_idx[v.id] for v in f.vertices()])
    return positions, faces


def _fan_triangulate(faces: Sequence[Sequence[int]]) -> List[Tuple[int, int, int]]:
    tris: List[Tuple[int, int, int]] = []
    for ring in faces:
        for j in range(1, len(ring) - 1):
            tris.append((ring[0], ring[j], ring[j + 1]))
    return tris


# ─────────────────────────────────────────────────────────────────────────────
# Float-op registry (single source of truth = existing implementations)
# ─────────────────────────────────────────────────────────────────────────────

def _sta(mesh):
    stellate_all(mesh)
    return mesh


# opcode -> callable
_LINEAR_FLOAT_OPS: Dict[str, Callable] = {
    "CC":    catmull_clark,
    "DUAL":  dual,
    "DS":    doo_sabin,
    "STA":   _sta,
    "SIMP":  simplest_subdivide,
    "VC":    vertex_cutting,
    "LOOP":  loop_subdivide,
    "SQRT3": sqrt3_subdivide,
    "HONEY": honeycomb_subdivide,
    "CCUT":  corner_cutting,
    "LSTYLE": loop_style_subdivide,
    "PENT":  pentagonal_subdivide,
    "PENT2": pentagonal2_subdivide,
    "D1264": dual1264_subdivide,
    "ROOT4": root4_subdivide,
    "CHKB":  checkerboard_remesh,
    "DSBC":  ds_bc_new_subdivide,
}

LINEAR_OPS: Tuple[str, ...] = tuple(_LINEAR_FLOAT_OPS.keys())
NONLINEAR_OPS: Tuple[str, ...] = (
    "CRUST", "STAR", "FRAC", "DOME",
    "EXTRUDE_FACE", "STELLATE", "SUBDIVIDE_EDGE", "SUBDIVIDE_FACE",
)


# ─────────────────────────────────────────────────────────────────────────────
# DiffOp
# ─────────────────────────────────────────────────────────────────────────────

class DiffOp:
    """
    One operator application with frozen topology and a differentiable
    position map.
    """

    def __init__(self, name: str, n_in: int, n_out: int,
                 faces: List[List[int]]):
        self.name = name
        self.n_in = n_in
        self.n_out = n_out
        self.faces = faces
        self._mat_cache: Dict[Tuple, torch.Tensor] = {}
        # linear payload
        self._idx: Optional[List[Tuple[int, int]]] = None
        self._val: Optional[List[float]] = None
        # nonlinear payload
        self._apply_fn: Optional[Callable] = None

    # -- linear path ------------------------------------------------------
    def _matrix(self, dtype, device) -> torch.Tensor:
        key = (dtype, device)
        mat = self._mat_cache.get(key)
        if mat is None:
            rows = [i for i, _ in self._idx]
            cols = [j for _, j in self._idx]
            mat = torch.sparse_coo_tensor(
                torch.tensor([rows, cols], dtype=torch.long, device=device),
                torch.tensor(self._val, dtype=dtype, device=device),
                size=(self.n_out, self.n_in),
            ).coalesce()
            self._mat_cache[key] = mat
        return mat

    def apply(self, verts: torch.Tensor) -> torch.Tensor:
        """Map input positions [n_in, 3] to output positions [n_out, 3]."""
        if verts.shape != (self.n_in, 3):
            raise ValueError(
                f"{self.name}: expected verts of shape ({self.n_in}, 3), "
                f"got {tuple(verts.shape)}")
        if self._apply_fn is not None:
            return self._apply_fn(verts)
        return torch.sparse.mm(self._matrix(verts.dtype, verts.device), verts)


# ─────────────────────────────────────────────────────────────────────────────
# Tracing (linear ops)
# ─────────────────────────────────────────────────────────────────────────────

_CONST_TOL = 1e-12


def _trace_linear(name: str, n_verts: int, faces: List[List[int]],
                  params: Dict) -> DiffOp:
    fn = _LINEAR_FLOAT_OPS[name]

    lin_positions = [(_Lin({i: 1.0}), _Lin({i: 1.0}), _Lin({i: 1.0}))
                     for i in range(n_verts)]
    mesh = _build_mesh(lin_positions, [list(r) for r in faces])
    out = fn(mesh, **params)
    if out is None:
        out = mesh

    vid_to_idx: Dict[int, int] = {}
    idx: List[Tuple[int, int]] = []
    val: List[float] = []
    for row, v in enumerate(out.vertices.values()):
        vid_to_idx[v.id] = row
        lx = v.x
        if not isinstance(lx, _Lin):        # pragma: no cover
            raise _NonLinearTrace(
                f"{name}: output coordinate is not symbolic ({type(lx)})")
        if abs(lx.k) > _CONST_TOL:
            raise _NonLinearTrace(
                f"{name}: affine constant {lx.k} in output — not linear")
        if v.y.c != lx.c or v.z.c != lx.c:
            raise _NonLinearTrace(
                f"{name}: channel-dependent weights — not a per-channel "
                "linear operator")
        for j, w in lx.c.items():
            if w != 0.0:
                idx.append((row, j))
                val.append(w)

    out_faces = [[vid_to_idx[v.id] for v in f.vertices()]
                 for f in out.faces.values()]

    op = DiffOp(name, n_verts, len(out.vertices), out_faces)
    op._idx, op._val = idx, val
    return op


# ─────────────────────────────────────────────────────────────────────────────
# Shared torch helpers (Newell normals, centroid, edge length)
# ─────────────────────────────────────────────────────────────────────────────

_EPS = 1e-12


def _newell_face_normals(verts: torch.Tensor,
                         faces: List[List[int]]) -> torch.Tensor:
    """
    Newell normal per face (normalized), matching Face.normal() including
    its degenerate fallback direction (0, 0, 1).
    Returns [F, 3].
    """
    normals = []
    for ring in faces:
        p = verts[torch.tensor(ring, dtype=torch.long, device=verts.device)]
        q = torch.roll(p, shifts=-1, dims=0)
        n = torch.stack([
            torch.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
            torch.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
            torch.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
        ])
        length = torch.linalg.vector_norm(n)
        if length.detach().item() < _EPS:
            n = torch.tensor([0.0, 0.0, 1.0], dtype=verts.dtype,
                             device=verts.device)
        else:
            n = n / length
        normals.append(n)
    return torch.stack(normals)


def _face_centroids(verts: torch.Tensor,
                    faces: List[List[int]]) -> torch.Tensor:
    """Centroid per face [F, 3]."""
    cs = []
    for ring in faces:
        cs.append(verts[ring].mean(dim=0))
    return torch.stack(cs)


# ─────────────────────────────────────────────────────────────────────────────
# CRUST (nonlinear, torch)
# ─────────────────────────────────────────────────────────────────────────────

def _make_crust_op(n_verts: int, faces: List[List[int]],
                   thickness) -> DiffOp:
    incident: List[List[int]] = [[] for _ in range(n_verts)]
    for fi, ring in enumerate(faces):
        for vi in ring:
            incident[vi].append(fi)

    out_faces = [list(r) for r in faces] + \
                [[vi + n_verts for vi in reversed(r)] for r in faces]

    def apply_fn(verts: torch.Tensor) -> torch.Tensor:
        t = thickness
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(float(t), dtype=verts.dtype, device=verts.device)
        fn = _newell_face_normals(verts, faces)
        vn = torch.stack([fn[incident[i]].sum(dim=0)
                          for i in range(n_verts)])
        mag = torch.linalg.vector_norm(vn, dim=1, keepdim=True)
        vn = vn / torch.clamp(mag, min=_EPS)
        inner = verts - t * vn
        return torch.cat([verts, inner], dim=0)

    op = DiffOp("CRUST", n_verts, 2 * n_verts, out_faces)
    op._apply_fn = apply_fn
    return op


# ─────────────────────────────────────────────────────────────────────────────
# EXTRUDE_FACE (nonlinear: face normal displacement)
# ─────────────────────────────────────────────────────────────────────────────

def _make_extrude_face_op(n_verts: int, faces: List[List[int]],
                          face_idx: int, dist) -> DiffOp:
    """
    Differentiable extrude_face on faces[face_idx].

    Runs the float path to get the output topology, then builds a torch
    function: old verts pass through; new verts = old + dist * normal.
    """
    # Run float for topology
    positions = [(0.0, 0.0, 0.0)] * n_verts
    mesh = _build_mesh(positions, [list(r) for r in faces])
    face_list = list(mesh.faces.values())
    target_face = face_list[face_idx]
    ring_vids = [v.id for v in target_face.vertices()]

    extrude_face(mesh, target_face, dist=1.0)
    out_pos, out_faces = mesh_to_arrays(mesh)
    n_out = len(out_pos)

    # Identify the vertex correspondence: original verts keep their
    # positions; new verts (n_out - n_verts of them) are offset copies
    # of the face ring vertices.
    # The extrude creates n new verts (one per ring vertex),
    # in the same order as the ring.
    vid_to_idx_before: Dict[int, int] = {}
    for i, v in enumerate(list(mesh.vertices.values())[:n_verts]):
        vid_to_idx_before[v.id] = i

    # The face ring vertex indices in our 0-based numbering:
    ring_indices: List[int] = []
    orig_verts = list(mesh.vertices.values())
    vid_to_idx_all: Dict[int, int] = {}
    for i, v in enumerate(orig_verts):
        vid_to_idx_all[v.id] = i

    # extrude_face creates new verts at indices n_verts .. n_out-1
    # They correspond 1:1 to the ring vertices (same order).
    # We need the input vertex indices that map to the ring.
    # Ring was face.vertices() before extrude. Build from the face ring.
    mesh2 = _build_mesh([(0.0, 0.0, 0.0)] * n_verts,
                        [list(r) for r in faces])
    face2 = list(mesh2.faces.values())[face_idx]
    ring_in_indices = [
        list(mesh2.vertices.keys()).index(v.id)
        for v in face2.vertices()
    ]

    # After extrude, vertex layout in the mesh is: first n_verts originals
    # (same order as input), then n_new = len(ring) new verts.
    n_new = n_out - n_verts

    def apply_fn(verts: torch.Tensor) -> torch.Tensor:
        d = dist
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(float(d), dtype=verts.dtype, device=verts.device)
        # Face normal from ring vertices
        ring_v = verts[ring_in_indices]
        q = torch.roll(ring_v, shifts=-1, dims=0)
        normal = torch.stack([
            torch.sum((ring_v[:, 1] - q[:, 1]) * (ring_v[:, 2] + q[:, 2])),
            torch.sum((ring_v[:, 2] - q[:, 2]) * (ring_v[:, 0] + q[:, 0])),
            torch.sum((ring_v[:, 0] - q[:, 0]) * (ring_v[:, 1] + q[:, 1])),
        ])
        nrm = torch.linalg.vector_norm(normal)
        if nrm.detach().item() < _EPS:
            normal = torch.tensor([0.0, 0.0, 1.0], dtype=verts.dtype,
                                  device=verts.device)
        else:
            normal = normal / nrm

        disp = d * normal  # [3]
        new_verts = verts[ring_in_indices] + disp.unsqueeze(0)
        return torch.cat([verts, new_verts], dim=0)

    op = DiffOp("EXTRUDE_FACE", n_verts, n_out, out_faces)
    op._apply_fn = apply_fn
    return op


# ─────────────────────────────────────────────────────────────────────────────
# STELLATE (nonlinear: centroid, optionally + dist * normal)
# ─────────────────────────────────────────────────────────────────────────────

def _make_stellate_op(n_verts: int, faces: List[List[int]],
                      face_idx: int, dist) -> DiffOp:
    """Stellate faces[face_idx]: new apex = centroid + dist * normal."""
    positions = [(0.0, 0.0, 0.0)] * n_verts
    mesh = _build_mesh(positions, [list(r) for r in faces])
    face_list = list(mesh.faces.values())
    target_face = face_list[face_idx]
    stellate(mesh, target_face)
    out_pos, out_faces = mesh_to_arrays(mesh)
    n_out = len(out_pos)

    ring_in_indices = list(faces[face_idx])

    def apply_fn(verts: torch.Tensor) -> torch.Tensor:
        ring_v = verts[ring_in_indices]
        centroid = ring_v.mean(dim=0)
        d = dist
        if not isinstance(d, torch.Tensor):
            d = torch.tensor(float(d), dtype=verts.dtype, device=verts.device)
        if float(d) != 0.0 or isinstance(dist, torch.Tensor):
            q = torch.roll(ring_v, shifts=-1, dims=0)
            normal = torch.stack([
                torch.sum((ring_v[:, 1] - q[:, 1]) * (ring_v[:, 2] + q[:, 2])),
                torch.sum((ring_v[:, 2] - q[:, 2]) * (ring_v[:, 0] + q[:, 0])),
                torch.sum((ring_v[:, 0] - q[:, 0]) * (ring_v[:, 1] + q[:, 1])),
            ])
            nrm = torch.linalg.vector_norm(normal)
            if nrm.detach().item() < _EPS:
                normal = torch.tensor([0.0, 0.0, 1.0], dtype=verts.dtype,
                                      device=verts.device)
            else:
                normal = normal / nrm
            apex = centroid + d * normal
        else:
            apex = centroid
        return torch.cat([verts, apex.unsqueeze(0)], dim=0)

    op = DiffOp("STELLATE", n_verts, n_out, out_faces)
    op._apply_fn = apply_fn
    return op


# ─────────────────────────────────────────────────────────────────────────────
# SUBDIVIDE_EDGE (linear: midpoint = average of endpoints)
# ─────────────────────────────────────────────────────────────────────────────

def _make_subdivide_edge_op(n_verts: int, faces: List[List[int]],
                            edge_verts: Tuple[int, int]) -> DiffOp:
    """
    Split the edge between edge_verts[0] and edge_verts[1] at its midpoint.
    """
    positions = [(0.0, 0.0, 0.0)] * n_verts
    mesh = _build_mesh(positions, [list(r) for r in faces])
    v0_id = list(mesh.vertices.values())[edge_verts[0]].id
    v1_id = list(mesh.vertices.values())[edge_verts[1]].id
    edge = mesh.find_edge(mesh.vertices[v0_id], mesh.vertices[v1_id])
    if edge is None:
        raise ValueError(f"No edge between vertices {edge_verts}")
    subdivide_edge(mesh, edge)
    out_pos, out_faces = mesh_to_arrays(mesh)
    n_out = len(out_pos)

    vi0, vi1 = edge_verts

    def apply_fn(verts: torch.Tensor) -> torch.Tensor:
        mid = (verts[vi0] + verts[vi1]) / 2.0
        return torch.cat([verts, mid.unsqueeze(0)], dim=0)

    op = DiffOp("SUBDIVIDE_EDGE", n_verts, n_out, out_faces)
    op._apply_fn = apply_fn
    return op


# ─────────────────────────────────────────────────────────────────────────────
# SUBDIVIDE_FACE (= stellate at dist=0; centroid)
# ─────────────────────────────────────────────────────────────────────────────

def _make_subdivide_face_op(n_verts: int, faces: List[List[int]],
                            face_idx: int) -> DiffOp:
    return _make_stellate_op(n_verts, faces, face_idx, dist=0.0)


# ─────────────────────────────────────────────────────────────────────────────
# STAR (nonlinear: stellate_all × 2 + offset · original face normals)
# ─────────────────────────────────────────────────────────────────────────────

def _make_star_op(n_verts: int, faces: List[List[int]],
                  offset) -> DiffOp:
    """
    Star subdivision = stellate_all twice + first-round apexes displaced by
    offset along original face normals.

    Strategy: run STA once via linear trace to get the intermediate
    topology + sparse matrix W1.  Then STA again via trace → W2.
    The position map is W2 @ (W1 @ verts + corrections), where
    corrections are nonzero only at the first-round apex rows
    (offset · face_normal).
    """
    # STA round 1 topology
    sta1 = _trace_linear("STA", n_verts, faces, {})
    mid_faces = sta1.faces
    n_mid = sta1.n_out

    # STA round 2 topology
    sta2 = _trace_linear("STA", n_mid, mid_faces, {})
    out_faces = sta2.faces
    n_out = sta2.n_out

    # First-round apex indices: STA adds one vertex per face at the END
    # of the vertex list (centroids), so apexes are indices n_verts .. n_mid-1.
    # They correspond 1:1 to faces[0], faces[1], ...
    apex_start = n_verts
    n_faces_orig = len(faces)

    # The round-1 apexes survive into the final mesh (STA round 2 only
    # adds new apexes, it doesn't move existing vertices — their rows in
    # W2 are identity).  The float code displaces them AFTER both rounds:
    #   stellate_all(mesh)       # round 1
    #   stellate_all(mesh)       # round 2
    #   apex.pos += offset * n   # displace round-1 apexes in place
    # So: final = W2 @ (W1 @ verts) + correction, where correction is
    # nonzero only at the round-1-apex rows of the FINAL vertex array.
    # We need to find where round-1 apexes land in the final layout.

    # Round-1 apexes are at indices apex_start..apex_start+n_faces_orig-1
    # in the intermediate (post-STA1) mesh.  STA2 preserves them at the
    # same row indices in its output (they are original vertices of the
    # STA2 input, and STA2 doesn't move originals — they keep their
    # positions via identity rows in W2).

    def apply_fn(verts: torch.Tensor) -> torch.Tensor:
        o = offset
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(float(o), dtype=verts.dtype, device=verts.device)

        # STA round 1 then round 2 (both linear)
        mid = sta1.apply(verts)
        out = sta2.apply(mid)

        # Displace round-1 apexes in the FINAL output
        if o.detach().item() != 0.0 or isinstance(offset, torch.Tensor):
            fn = _newell_face_normals(verts, faces)
            corrections = o * fn
            out = out.clone()
            out[apex_start:apex_start + n_faces_orig] = \
                out[apex_start:apex_start + n_faces_orig] + corrections

        return out

    op = DiffOp("STAR", n_verts, n_out, out_faces)
    op._apply_fn = apply_fn
    return op


# ─────────────────────────────────────────────────────────────────────────────
# FRACTAL (nonlinear: loop_style linear + apex = centroid + h·normal)
# ─────────────────────────────────────────────────────────────────────────────

def _make_fractal_op(n_verts: int, faces: List[List[int]],
                     offset) -> DiffOp:
    """
    Fractal subdivision = loop_style + stellate central polygon.

    Strategy: trace LSTYLE for the linear part (positions of old verts +
    midpoints), then compute apex positions as
    centroid + h * normal where h = offset * sqrt(max(L2²-L1², 0)).
    """
    # Run float path for topology
    positions_dummy = [(0.0, 0.0, 0.0)] * n_verts
    mesh_f = _build_mesh(positions_dummy, [list(r) for r in faces])
    frac_out = fractal_subdivide(mesh_f, offset=1.0)
    out_pos, out_faces = mesh_to_arrays(frac_out)
    n_out = len(out_pos)

    # Trace LSTYLE for the shared vertex layout (positions of V + edge midpoints)
    lstyle_op = _trace_linear("LSTYLE", n_verts, faces, {"length": 1.0})
    n_lstyle = lstyle_op.n_out
    lstyle_faces = lstyle_op.faces

    # From the float path we know: fractal output has
    # V' = V + E + F vertices.  LSTYLE output has V + E vertices.
    # The extra F vertices are the face apexes.
    # apex_i sits at index n_lstyle + i in the output.

    # For each original face, we need:
    # 1. The ring vertex indices in the LSTYLE output (the midpoint d-gon)
    #    — these are the "central polygon" vertices.
    # 2. To compute the apex from the original face's vertices.

    # Build the edge midpoint index map for the original mesh
    mesh_ref = _build_mesh(positions_dummy, [list(r) for r in faces])
    orig_edges = list(mesh_ref.edges.values())
    orig_verts_list = list(mesh_ref.vertices.values())
    vid_to_idx_in: Dict[int, int] = {}
    for i, v in enumerate(orig_verts_list):
        vid_to_idx_in[v.id] = i
    eid_to_idx: Dict[int, int] = {}
    for i, e in enumerate(orig_edges):
        eid_to_idx[e.id] = len(orig_verts_list) + i

    # For each face: midpoint polygon ring (LSTYLE indices), and
    # face vertex indices (original, for normal/centroid)
    face_mid_rings: List[List[int]] = []     # indices into LSTYLE output
    face_vert_indices: List[List[int]] = []  # indices into input
    # edge pair indices for L2 computation
    face_edge_pairs: List[Tuple[int, int]] = []  # (eid0, eid1) in LSTYLE space
    # vertex pairs for L1 computation
    face_vert_pairs: List[Tuple[int, int]] = []

    for f in mesh_ref.iter_faces():
        hes = f.halfedges()
        mid_ring = [eid_to_idx[he.edge.id] for he in hes]
        face_mid_rings.append(mid_ring)
        verts_f = [vid_to_idx_in[he.origin.id] for he in hes]
        face_vert_indices.append(verts_f)
        # L2: distance between first two consecutive midpoints
        face_edge_pairs.append((mid_ring[0], mid_ring[1]))
        # L1: half distance between v[0] and v[d//2]
        d = len(verts_f)
        face_vert_pairs.append((verts_f[0], verts_f[d // 2]))

    n_faces_orig = len(faces)

    def apply_fn(verts: torch.Tensor) -> torch.Tensor:
        o = offset
        if not isinstance(o, torch.Tensor):
            o = torch.tensor(float(o), dtype=verts.dtype, device=verts.device)

        # Compute LSTYLE positions (linear)
        lstyle_verts = lstyle_op.apply(verts)  # [n_lstyle, 3]

        # Compute apex per original face
        fn = _newell_face_normals(verts, faces)         # [n_faces_orig, 3]
        centroids = _face_centroids(verts, faces)       # [n_faces_orig, 3]

        apexes = []
        for fi in range(n_faces_orig):
            # L2: distance between consecutive midpoints in lstyle
            m0 = lstyle_verts[face_edge_pairs[fi][0]]
            m1 = lstyle_verts[face_edge_pairs[fi][1]]
            L2_sq = ((m1 - m0) ** 2).sum()
            # L1: half distance between opposite corners
            va = verts[face_vert_pairs[fi][0]]
            vb = verts[face_vert_pairs[fi][1]]
            L1_sq = ((vb - va) ** 2).sum() / 4.0
            diff = torch.clamp(L2_sq - L1_sq, min=0.0)
            # sqrt(0) must produce 0 (matching float path) with finite
            # gradient.  Use sqrt(diff + eps) then zero out when diff==0.
            safe_sqrt = torch.sqrt(diff + _EPS)
            # Mask: when diff is effectively zero, h should be zero
            h = o * safe_sqrt * (diff > _EPS).to(verts.dtype)
            apexes.append(centroids[fi] + h * fn[fi])

        apex_tensor = torch.stack(apexes)
        return torch.cat([lstyle_verts, apex_tensor], dim=0)

    op = DiffOp("FRAC", n_verts, n_out, out_faces)
    op._apply_fn = apply_fn
    return op


# ─────────────────────────────────────────────────────────────────────────────
# DOME (nonlinear: quadrisect + 7x extrude with DS ring repositioning)
# ─────────────────────────────────────────────────────────────────────────────

def _make_dome_op(n_verts: int, faces: List[List[int]],
                  length_param, sf_param) -> DiffOp:
    """
    Dome subdivision.

    Strategy: run the full float dome to get output topology + reference
    positions for oracle verification. Then build a torch function that
    reconstructs positions step by step:
    1. Quadrisect edges (linear — midpoints of midpoints)
    2. For each original face, 7 rounds of extrude + DS ring repositioning
       (all expressible as face normals + centroids + linear mixing)
    """
    # Run float path for topology and reference
    positions_dummy = [(float(i), float(i * 2), float(i * 3))
                       for i in range(n_verts)]
    mesh_f = _build_mesh(positions_dummy, [list(r) for r in faces])
    dome_subdivide(mesh_f, length=1.0, sf=1.0)
    out_pos, out_faces = mesh_to_arrays(mesh_f)
    n_out = len(out_pos)

    # The dome is too complex for a step-by-step torch reconstruction in
    # a single pass. Instead, use the "run float path, match positions"
    # approach: run the float op with actual input positions, then build
    # a torch computation graph that reproduces the output.
    #
    # For the dome, we implement a full torch version that mirrors the
    # float path exactly.

    # Pre-compute the topology: for each original face, which vertices
    # are in its boundary after quadrisection, and how the 7 extrusion
    # rounds transform them.

    # Since dome is composed of operations we already handle (quadrisect =
    # linear subdivide, extrude = normal displacement, DS ring reposition =
    # linear mixing), we trace it as a compound operation.

    # Step 1: Quadrisection = subdivide each edge into 4.
    # Each edge midpoint = average of endpoints (linear).
    # After quadrisection each original d-gon becomes a 4d-gon.
    # This is equivalent to applying subdivide_edge 3x per original edge.

    # Build edge adjacency from input faces
    edge_set: List[Tuple[int, int]] = []   # ordered pairs
    edge_lookup: Dict[Tuple[int, int], int] = {}
    for ring in faces:
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            key = (min(a, b), max(a, b))
            if key not in edge_lookup:
                edge_lookup[key] = len(edge_set)
                edge_set.append((a, b))

    n_edges = len(edge_set)

    # Quadrisection: each edge a—b becomes a—q1—q2—q3—b with
    # q1 = (3a+b)/4, q2 = (a+b)/2, q3 = (a+3b)/4
    # New vertex indices: n_verts + 3*edge_idx + {0,1,2}
    n_quad = n_verts + 3 * n_edges

    # Build quadrisected face rings
    quad_faces: List[List[int]] = []
    for ring in faces:
        new_ring = []
        for k in range(len(ring)):
            a, b = ring[k], ring[(k + 1) % len(ring)]
            new_ring.append(a)
            key = (min(a, b), max(a, b))
            ei = edge_lookup[key]
            # q1, q2, q3 indices
            q1_idx = n_verts + 3 * ei
            q2_idx = n_verts + 3 * ei + 1
            q3_idx = n_verts + 3 * ei + 2
            if edge_set[ei][0] == a:  # same direction
                new_ring.extend([q1_idx, q2_idx, q3_idx])
            else:  # reversed
                new_ring.extend([q3_idx, q2_idx, q1_idx])
        quad_faces.append(new_ring)

    # Precompute per-original-face average edge length (at runtime, from
    # input verts — this is the "unit" for dome height)
    face_edge_lists: List[List[Tuple[int, int]]] = []
    for ring in faces:
        edges = []
        for k in range(len(ring)):
            edges.append((ring[k], ring[(k + 1) % len(ring)]))
        face_edge_lists.append(edges)

    n_faces_orig = len(faces)
    heights = _DOME_HEIGHTS
    scales = _DOME_SCALES

    def apply_fn(verts: torch.Tensor) -> torch.Tensor:
        lng = length_param
        if not isinstance(lng, torch.Tensor):
            lng = torch.tensor(float(lng), dtype=verts.dtype,
                               device=verts.device)
        s_f = sf_param
        if not isinstance(s_f, torch.Tensor):
            s_f = torch.tensor(float(s_f), dtype=verts.dtype,
                               device=verts.device)

        # Quadrisect: build all vertices
        quad_verts_list = [verts]  # start with originals
        q_new = []
        for (a, b) in edge_set:
            va, vb = verts[a], verts[b]
            q1 = 0.75 * va + 0.25 * vb
            q2 = 0.50 * va + 0.50 * vb
            q3 = 0.25 * va + 0.75 * vb
            q_new.extend([q1, q2, q3])
        all_verts = torch.cat([verts, torch.stack(q_new)], dim=0)  # [n_quad, 3]

        # Now perform 7 extrusion rounds per face.
        # We accumulate all created vertices in a list and track the
        # current face ring indices.
        extra_verts: List[torch.Tensor] = []
        current_rings: List[List[int]] = [list(r) for r in quad_faces]
        next_idx = n_quad  # next available vertex index

        for round_idx in range(7):
            h = heights[round_idx]
            s = scales[round_idx]
            new_rings: List[List[int]] = []

            for fi in range(n_faces_orig):
                ring = current_rings[fi]
                n = len(ring)

                # Get ring vertex positions
                # Vertices may be in all_verts or in extra_verts
                ring_pts = []
                for vi in ring:
                    if vi < all_verts.shape[0]:
                        ring_pts.append(all_verts[vi])
                    else:
                        ring_pts.append(extra_verts[vi - all_verts.shape[0]])
                ring_pts = torch.stack(ring_pts)  # [n, 3]

                # Face normal (Newell)
                q = torch.roll(ring_pts, shifts=-1, dims=0)
                normal = torch.stack([
                    torch.sum((ring_pts[:, 1] - q[:, 1]) *
                              (ring_pts[:, 2] + q[:, 2])),
                    torch.sum((ring_pts[:, 2] - q[:, 2]) *
                              (ring_pts[:, 0] + q[:, 0])),
                    torch.sum((ring_pts[:, 0] - q[:, 0]) *
                              (ring_pts[:, 1] + q[:, 1])),
                ])
                nrm = torch.linalg.vector_norm(normal)
                if nrm.detach().item() < _EPS:
                    normal = torch.tensor([0.0, 0.0, 1.0], dtype=verts.dtype,
                                          device=verts.device)
                else:
                    normal = normal / nrm

                # Average edge length for height unit
                edge_lens = []
                for fi2, edges in enumerate(face_edge_lists):
                    if fi2 == fi:
                        for (a, b) in edges:
                            edge_lens.append(
                                torch.linalg.vector_norm(verts[a] - verts[b]))
                unit = torch.stack(edge_lens).mean() if edge_lens else \
                    torch.tensor(1.0, dtype=verts.dtype, device=verts.device)

                # Extrude: new verts = old + h * length * unit * normal
                disp = h * lng * unit * normal
                new_pts = ring_pts + disp.unsqueeze(0)  # [n, 3]

                # DS ring repositioning: scale about centroid
                centroid = new_pts.mean(dim=0)
                scale = s * s_f
                # DS mask: ds_pt[k] = (p[k] + centroid + (p[k]+p[k-1])/2
                #                      + (p[k]+p[k+1])/2) / 4
                p_prev = torch.roll(new_pts, shifts=1, dims=0)
                p_next = torch.roll(new_pts, shifts=-1, dims=0)
                ds_pts = (new_pts + centroid.unsqueeze(0) +
                          (new_pts + p_prev) / 2 +
                          (new_pts + p_next) / 2) / 4.0
                # Scale about centroid
                repositioned = centroid.unsqueeze(0) + scale * (
                    ds_pts - centroid.unsqueeze(0))

                # Store new vertices
                new_ring_indices = []
                for k in range(n):
                    extra_verts.append(repositioned[k])
                    new_ring_indices.append(next_idx)
                    next_idx += 1

                new_rings.append(new_ring_indices)

            current_rings = new_rings

        # Assemble final vertex tensor
        if extra_verts:
            return torch.cat([all_verts, torch.stack(extra_verts)], dim=0)
        return all_verts

    op = DiffOp("DOME", n_verts, n_out, out_faces)
    op._apply_fn = apply_fn
    return op


# ─────────────────────────────────────────────────────────────────────────────
# HDL (add_handle) — topology-only, identity on vertex positions
# ─────────────────────────────────────────────────────────────────────────────

def _make_hdl_op(n_verts: int, faces: List[List[int]],
                 face1_ord: int = 0, face2_ord: int = 1) -> DiffOp:
    """
    add_handle connects two faces with a tube (genus +1).

    No new vertices are created — only face connectivity changes.
    The DiffOp is therefore an identity map on positions, with updated faces.
    """
    # Build a temporary DLFL mesh to run the float add_handle
    positions = [(0.0, 0.0, 0.0)] * n_verts  # dummy positions; HDL doesn't use them
    mesh = _build_mesh(positions, faces)

    # Look up faces by ordinal
    face_list = list(mesh.faces.values())
    if face1_ord >= len(face_list) or face2_ord >= len(face_list):
        raise ValueError(
            f"HDL face ordinals ({face1_ord}, {face2_ord}) out of range "
            f"(mesh has {len(face_list)} faces)")
    f1 = face_list[face1_ord]
    f2 = face_list[face2_ord]

    add_handle(mesh, f1, f2)

    # Extract updated faces (vertex count unchanged)
    _, out_faces = mesh_to_arrays(mesh)
    n_out = n_verts  # HDL never creates new vertices

    # Identity map: output positions = input positions
    op = DiffOp("HDL", n_verts, n_out, out_faces)
    op._apply_fn = lambda verts: verts  # identity — pure topology change
    return op


# ─────────────────────────────────────────────────────────────────────────────
# Public constructors
# ─────────────────────────────────────────────────────────────────────────────

def trace_op(name: str, positions_or_n, faces: List[List[int]],
             **params) -> DiffOp:
    """
    Build a DiffOp for operator *name* applied to the topology given by
    *faces* (polygon rings over ``n`` input vertices).

    positions_or_n : int vertex count, or a positions list (len used).
    params         : the operator's usual keyword parameters.  For
                     nonlinear ops, continuous parameters may be torch
                     scalar tensors (differentiable).
    """
    n = (positions_or_n if isinstance(positions_or_n, int)
         else len(positions_or_n))
    name = name.upper()
    if name in _LINEAR_FLOAT_OPS:
        return _trace_linear(name, n, faces, params)
    if name == "CRUST":
        return _make_crust_op(n, faces, params.get("thickness", 0.1))
    if name == "STAR":
        return _make_star_op(n, faces, params.get("offset", 0.0))
    if name == "FRAC":
        return _make_fractal_op(n, faces, params.get("offset", 1.0))
    if name == "DOME":
        return _make_dome_op(n, faces,
                             params.get("length", 1.0),
                             params.get("sf", 1.0))
    if name == "EXTRUDE_FACE":
        return _make_extrude_face_op(n, faces,
                                     params.get("face_idx", 0),
                                     params.get("dist", 1.0))
    if name == "STELLATE":
        return _make_stellate_op(n, faces,
                                 params.get("face_idx", 0),
                                 params.get("dist", 0.0))
    if name == "SUBDIVIDE_EDGE":
        ev = params.get("edge_verts")
        if ev is None:
            raise ValueError("SUBDIVIDE_EDGE requires edge_verts=(i,j)")
        return _make_subdivide_edge_op(n, faces, tuple(ev))
    if name == "SUBDIVIDE_FACE":
        return _make_subdivide_face_op(n, faces, params.get("face_idx", 0))
    if name == "HDL":
        return _make_hdl_op(n, faces,
                            params.get("face1_ord", 0),
                            params.get("face2_ord", 1))
    if name in ("IE", "DE"):
        # IE/DE are topology-only ops; not yet implemented in DiffSequence.
        raise ValueError(
            f"IE/DE not yet supported in DiffSequence "
            f"(topology-only ops require mesh rebuild)")
    raise ValueError(
        f"unsupported op {name!r}; linear={LINEAR_OPS}, "
        f"nonlinear={NONLINEAR_OPS}, topology=('HDL',)")


_BASES: Dict[str, Callable[[], DLFLMesh]] = {
    "cube": make_cube,
    "tetrahedron": make_tetrahedron,
    "icosahedron": make_icosahedron,
}


class DiffSequence:
    """
    A fixed operator sequence applied from a base primitive, composed into
    one differentiable map ``base verts [V0,3] -> final verts [Vn,3]``.

    >>> seq = DiffSequence("cube").append("DS").append("CC")
    >>> final = seq.forward()          # differentiable w.r.t. seq.verts0
    >>> tris  = seq.triangles()        # for nvdiffrast / geometry_optimizer
    """

    def __init__(self, base: str = "icosahedron", *,
                 dtype: torch.dtype = torch.float64,
                 device: str = "cpu",
                 requires_grad: bool = True):
        if isinstance(base, str):
            positions, faces = mesh_to_arrays(_BASES[base]())
        else:                       # (positions, faces) tuple
            positions, faces = base
        self.verts0 = torch.tensor(positions, dtype=dtype, device=device,
                                   requires_grad=requires_grad)
        self._faces: List[List[int]] = [list(r) for r in faces]
        self.ops: List[DiffOp] = []

    @property
    def faces(self) -> List[List[int]]:
        """Current (final) polygon rings."""
        return self._faces

    def append(self, name: str, **params) -> "DiffSequence":
        n_in = self.ops[-1].n_out if self.ops else self.verts0.shape[0]
        op = trace_op(name, n_in, self._faces, **params)
        self.ops.append(op)
        self._faces = op.faces
        return self

    def forward(self, verts0: Optional[torch.Tensor] = None) -> torch.Tensor:
        v = self.verts0 if verts0 is None else verts0
        for op in self.ops:
            v = op.apply(v)
        return v

    __call__ = forward

    def triangles(self, device: str = "cpu") -> torch.Tensor:
        """Fan-triangulated face indices as an int64 tensor [T, 3]."""
        return torch.tensor(_fan_triangulate(self._faces),
                            dtype=torch.long, device=device)
