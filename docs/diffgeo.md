# Differentiable Geometry (`topmod/diffgeo.py`)

PyTorch-differentiable geometry for **all 29 TopMod operators**. Topology
stays discrete (decided by the operator sequence, no gradient); vertex
positions become a differentiable torch function of the base-primitive
positions.

## How it works

**Linear operators (17) — symbolic trace.** The existing float
implementation is re-run with every coordinate replaced by a symbolic
linear-combination object; the output coordinates literally are the rows of
a sparse weight matrix `W`, so `new_verts = W @ old_verts`
(`torch.sparse.mm`). The float implementation stays the single source of
truth — no geometry rule is duplicated. Any nonlinear coordinate use
(products, `sqrt`, comparisons, `float()` casts) raises
`_NonLinearTrace` during tracing, so a wrongly classified operator cannot
silently produce wrong gradients.

**Nonlinear operators (7) — dedicated torch implementations.** Each uses
Newell face normals and/or edge lengths with eps-guarded normalization and
sqrt. Continuous parameters (`dist`, `offset`, `thickness`, `length`, `sf`)
may be passed as torch scalar tensors and receive gradients.

**Decomposition strategies used:**
- **STAR**: linear skeleton (two `STA` traces) + post-hoc offset · normal
  correction on first-round apexes
- **FRAC**: linear `LSTYLE` trace for shared vertices + torch apex
  positions (centroid + h · normal, h = offset · √max(L2²−L1², 0))
- **DOME**: torch quadrisection (linear midpoints) + 7 rounds of torch
  extrude (normal displacement) + DS ring repositioning (linear mix +
  centroid scaling)

## Supported operators

| Group | Ops | Gradients w.r.t. |
|---|---|---|
| Linear (17) | CC, DUAL, DS, STA, SIMP, VC, LOOP, SQRT3, HONEY, CCUT, LSTYLE, PENT, PENT2, D1264, ROOT4, CHKB, DSBC | input vertex positions (op parameters baked into traced weights as constants) |
| Nonlinear (7) | CRUST, STAR, FRAC, DOME, EXTRUDE_FACE, STELLATE, SUBDIVIDE_EDGE, SUBDIVIDE_FACE | input vertex positions **and** continuous parameters (`thickness`, `offset`, `dist`, `length`, `sf`) |
| Identity (3) | IE, DE, HDL | pure topology, no coordinates created |
| Free parameter (1) | CV | position is itself a leaf tensor |
| No geometry (1) | delete_vertex | — |

**100% coverage — every operator in the vocabulary is differentiable.**

Parameter differentiability for linear ops (offset/alpha/sf/...) would
require a dual-number trace — planned, not yet implemented.

## API

```python
import torch
from topmod.diffgeo import DiffSequence, trace_op, mesh_to_arrays

# Sequence from a base primitive (cube | tetrahedron | icosahedron)
seq = DiffSequence("cube").append("DS").append("CC") \
                          .append("CRUST", thickness=0.1)
final_verts = seq.forward()          # [Vn, 3], differentiable w.r.t. seq.verts0
tris        = seq.triangles()        # int64 [T, 3], fan-triangulated
loss = (final_verts ** 2).sum()
loss.backward()                      # gradients reach seq.verts0

# Single op on explicit topology
positions, faces = mesh_to_arrays(some_mesh)
op  = trace_op("CC", len(positions), faces)
out = op.apply(torch.tensor(positions, dtype=torch.float64))  # [V_out, 3]

# Nonlinear ops with differentiable parameters
op = trace_op("EXTRUDE_FACE", n_verts, faces, face_idx=0,
              dist=torch.tensor(0.6, requires_grad=True))
op = trace_op("STAR", n_verts, faces,
              offset=torch.tensor(0.15, requires_grad=True))
op = trace_op("DOME", n_verts, faces,
              length=torch.tensor(1.0, requires_grad=True),
              sf=torch.tensor(1.0, requires_grad=True))
```

`seq.triangles()` + `seq.forward()` plug directly into
`pipeline/geometry_optimizer.py` (nvdiffrast silhouette fitting): optimize
`seq.verts0` (8–12 numbers for a primitive) instead of thousands of final
vertices — drastically fewer degrees of freedom, natural regularization.

## Correctness contract (enforced by `tests/test_diffgeo.py`)

- **Oracle parity**: torch forward == existing float implementation
  (positions to 1e-9 AND identical face rings, or position-matched for
  in-place ops like DOME) for every op on cube/tetrahedron/icosahedron,
  including non-default parameters.
- **Topology-only trace**: a matrix traced once applies correctly to any
  positions with the same topology.
- **Gradients**: `torch.autograd.gradcheck` for all 17 linear ops;
  gradient-flow tests for all nonlinear ops; end-to-end sequence gradient
  flow to base vertices; parameter gradients for CRUST thickness,
  EXTRUDE_FACE dist, etc.

All CPU, float64. Run:

```bash
python3 -m pytest tests/test_diffgeo.py -q
```

## Design notes

- `torch` is imported only inside `topmod/diffgeo.py`; the topmod core
  remains torch-free (optional dependency).
- `dlfl.Vertex` passes non-numeric coordinate objects through without
  `float()` coercion (numbers are still coerced) — the one-line core change
  that enables tracing, invisible to all normal use.
- Traced matrices are cached per (dtype, device) inside each `DiffOp`.
- Vertex-index alignment: trace and float reference both build their input
  via `_build_mesh` from identical (positions, faces) arrays, so the
  deterministic construction order guarantees index-aligned comparison.
  For in-place ops (DOME), positions are verified up to permutation.
