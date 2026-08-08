"""
topmod/diffgeo.py — differentiable (PyTorch) geometry for TopMod operators.

Topology stays discrete (which vertices/edges/faces exist and how they are
connected is decided by the operator and carries no gradient); geometry
(where the vertices are placed) becomes a differentiable torch function of
the input vertex positions.

Two mechanisms
--------------
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

2. **Nonlinear operators** (currently: crust): dedicated torch
   implementation (Newell normals, eps-guarded normalization), verified
   against the float path by the test suite.  The continuous parameter
   (``thickness``) may be passed as a torch scalar tensor and receives
   gradients.

Scope notes
-----------
- For linear operators the continuous parameters (offset, alpha, sf, ...)
  are baked into the traced weights as constants: gradients flow through
  vertex *positions*, not through those parameters.  (Parameter
  differentiability for linear ops = phase 2, dual-number trace.)
- STAR / FRAC / DOME use face normals / edge lengths and are not yet
  covered (phase 2 custom torch implementations).
- torch is imported only inside this module; the topmod core stays
  torch-free.

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
    create_crust,
)
from .high_level_ops import stellate_all


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


# opcode -> (callable, returns_new_mesh)
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
NONLINEAR_OPS: Tuple[str, ...] = ("CRUST",)


# ─────────────────────────────────────────────────────────────────────────────
# DiffOp
# ─────────────────────────────────────────────────────────────────────────────

class DiffOp:
    """
    One operator application with frozen topology and a differentiable
    position map.

    Attributes
    ----------
    name     : opcode
    n_in     : input vertex count
    n_out    : output vertex count
    faces    : output polygon rings (list of vertex-index lists)
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
        if not isinstance(lx, _Lin):        # pragma: no cover — safety net
            raise _NonLinearTrace(
                f"{name}: output coordinate is not symbolic ({type(lx)})")
        if abs(lx.k) > _CONST_TOL:
            raise _NonLinearTrace(
                f"{name}: affine constant {lx.k} in output — not linear")
        # sanity: identical weights across the three channels
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
# CRUST (nonlinear, torch)
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
        q = torch.roll(p, shifts=-1, dims=0)          # next vertex
        n = torch.stack([
            torch.sum((p[:, 1] - q[:, 1]) * (p[:, 2] + q[:, 2])),
            torch.sum((p[:, 2] - q[:, 2]) * (p[:, 0] + q[:, 0])),
            torch.sum((p[:, 0] - q[:, 0]) * (p[:, 1] + q[:, 1])),
        ])
        length = torch.linalg.vector_norm(n)
        if length.detach().item() < 1e-12:
            n = torch.tensor([0.0, 0.0, 1.0], dtype=verts.dtype,
                             device=verts.device)
        else:
            n = n / length
        normals.append(n)
    return torch.stack(normals)


def _make_crust_op(n_verts: int, faces: List[List[int]],
                   thickness) -> DiffOp:
    # incident faces per vertex (order irrelevant: they are summed)
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
        fn = _newell_face_normals(verts, faces)                 # [F, 3]
        vn = torch.stack([fn[incident[i]].sum(dim=0)
                          for i in range(n_verts)])             # [V, 3]
        mag = torch.linalg.vector_norm(vn, dim=1, keepdim=True)
        # float path: normalize only when mag > 1e-12, else leave the
        # (near-zero) sum as-is.  Smooth eps-guarded equivalent:
        vn = vn / torch.clamp(mag, min=1e-12)
        inner = verts - t * vn
        return torch.cat([verts, inner], dim=0)

    op = DiffOp("CRUST", n_verts, 2 * n_verts, out_faces)
    op._apply_fn = apply_fn
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
    params         : the operator's usual keyword parameters.  For CRUST,
                     ``thickness`` may be a torch scalar tensor
                     (differentiable); for linear ops the parameters are
                     baked into the traced weights as constants.
    """
    n = (positions_or_n if isinstance(positions_or_n, int)
         else len(positions_or_n))
    name = name.upper()
    if name in _LINEAR_FLOAT_OPS:
        return _trace_linear(name, n, faces, params)
    if name == "CRUST":
        return _make_crust_op(n, faces, params.get("thickness", 0.1))
    raise ValueError(
        f"unsupported op {name!r}; linear={LINEAR_OPS}, "
        f"nonlinear={NONLINEAR_OPS}")


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
