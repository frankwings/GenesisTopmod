# Differentiable Geometry (`topmod/diffgeo.py`)

PyTorch-differentiable geometry for TopMod operators. Topology stays
discrete (decided by the operator sequence, no gradient); vertex positions
become a differentiable torch function of the base-primitive positions.

## How it works

**Linear operators — symbolic trace.** The existing float implementation is
re-run with every coordinate replaced by a symbolic linear-combination
object; the output coordinates literally are the rows of a sparse weight
matrix `W`, so `new_verts = W @ old_verts` (`torch.sparse.mm`). The float
implementation stays the single source of truth — no geometry rule is
duplicated. Any nonlinear coordinate use (products, `sqrt`, comparisons,
`float()` casts) raises `_NonLinearTrace` during tracing, so a wrongly
classified operator cannot silently produce wrong gradients.

**Nonlinear operators — dedicated torch implementation.** Currently CRUST:
Newell face normals + eps-guarded per-vertex normal averaging, verified
against the float path to 1e-9.

## Supported operators

| Group | Ops | Gradients w.r.t. |
|---|---|---|
| Linear (17) | CC, DUAL, DS, STA, SIMP, VC, LOOP, SQRT3, HONEY, CCUT, LSTYLE, PENT, PENT2, D1264, ROOT4, CHKB, DSBC | input vertex positions (op parameters are baked into the traced weights as constants) |
| Nonlinear (1) | CRUST | input vertex positions **and** `thickness` (pass a torch scalar tensor) |
| No geometry | IE, DE, HDL | identity — nothing to differentiate |
| Phase 2 | STAR, FRAC, DOME | need custom torch normal/length code (like CRUST) |

Parameter differentiability for linear ops (offset/alpha/sf/...) would
require a dual-number trace — planned, not implemented.

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
```

`seq.triangles()` + `seq.forward()` plug directly into
`pipeline/geometry_optimizer.py` (nvdiffrast silhouette fitting): optimize
`seq.verts0` (12–8 numbers for a primitive) instead of thousands of final
vertices — drastically fewer degrees of freedom, natural regularization.

## Correctness contract (enforced by `tests/test_diffgeo.py`)

- **Oracle parity**: torch forward == existing float implementation
  (positions to 1e-9 AND identical face rings) for every op on
  cube/tetrahedron/icosahedron, including non-default parameters.
- **Topology-only trace**: a matrix traced once applies correctly to any
  positions with the same topology.
- **Gradients**: `torch.autograd.gradcheck` for all 17 linear ops + CRUST;
  end-to-end sequence gradient flow to base vertices; CRUST `thickness`
  gradient.

All CPU, float64. Run:

```bash
python3 -m pytest tests/test_diffgeo.py -q
```

## Design notes

- `torch` is imported only inside `topmod/diffgeo.py`; the topmod core
  remains torch-free (optional dependency).
- `dlfl.Vertex` now passes non-numeric coordinate objects through without
  `float()` coercion (numbers are still coerced) — the one-line core change
  that enables tracing, invisible to all normal use.
- Traced matrices are cached per (dtype, device) inside each `DiffOp`.
- Vertex-index alignment: trace and float reference both build their input
  via `_build_mesh` from identical (positions, faces) arrays, so the
  deterministic construction order guarantees index-aligned comparison.
