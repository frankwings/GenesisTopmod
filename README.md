# GenesisTopmod

**Topology-Guaranteed Mesh Generation via TopMod Operators**

A pure-Python implementation of Dr. Ergun Akleman's TopMod topological mesh
theory — 29 mesh operators that guarantee valid orientable 2-manifold output
at every step. Includes a Blender addon, differentiable (PyTorch) geometry,
and an autoregressive tokenizer for AI mesh generation.

## Features

- **29 mesh operators** — 4 fundamental (Akleman & Chen 2003) + 6 high-level
  + 7 classic subdivision + 12 TopMod remeshing schemes — all with closed-form
  oracle tests
- **100% differentiable** — every operator has a PyTorch-differentiable
  position map (`topmod/diffgeo.py`); 17 via sparse-matrix symbolic trace,
  7 via dedicated torch implementations
- **Blender addon** — one-click install, 21 operators in Edit Mode with
  parameter sliders, Mesh menu + N-panel sidebar
- **Autoregressive tokenizer** — mesh ↔ integer token sequences for
  generative models; append-only vocabulary (backward compatible)
- **Pure Python core** — zero external dependencies (no NumPy required);
  torch is optional (only for `diffgeo.py`)

## Quick Start

```bash
git clone https://github.com/frankwings/GenesisTopmod.git
cd GenesisTopmod

# Run tests (381 core + 84 differentiable geometry = 465 total)
python3 -m pytest tests/ -q --ignore=tests/test_manifold_loss.py --ignore=tests/test_pipeline.py

# Try it
python3 -c "
from topmod import make_cube, catmull_clark, dual, honeycomb_subdivide
mesh = honeycomb_subdivide(make_cube())
print(f'Honeycomb cube: V={mesh.V()} E={mesh.E()} F={mesh.F()}')
"
```

## Blender Addon

### Download & Install

1. Download `blender_addon/topmod_blender.zip` from this repo
   ([direct link](https://github.com/frankwings/GenesisTopmod/raw/main/blender_addon/topmod_blender.zip))
2. Open Blender (3.6+)
3. **Edit → Preferences → Add-ons → Install...** → select the zip
4. Check the box next to **"Mesh: TopMod (DLFL Mesh Operators)"**

### Usage

1. Select a mesh object → **Tab** to enter Edit Mode
2. Open the sidebar: **N** → **TopMod** tab
3. Click any operator — parameters appear in the bottom-left popup (or **F9**)
4. **Ctrl+Z** to undo

### Available Blender Operators

| Category | Operators |
|---|---|
| High-Level | Stellate All |
| Classic Subdivision | Catmull-Clark, Dual, Doo-Sabin, Simplest, Vertex Cutting, Loop, √3 |
| TopMod Remeshing | Honeycomb, Star, Corner Cutting, Loop-Style, Fractal, Pentagonal, Pentagonal 2, Dual 12.6.4, Root-4, Checkerboard, DS BC-New, Dome |
| Structural | Create Crust |

See [`blender_addon/README.md`](blender_addon/README.md) for the full guide
with parameter details and tips.

## Operator Reference

Full documentation for all 29 operators with signatures, parameters,
closed-form oracles, differentiability details, and before/after
visualizations:

📄 [`docs/operators.md`](docs/operators.md)

### Quick Reference Table

| # | Operator | Token | Differentiable | Oracle (V', E', F') |
|---|---|---|---|---|
| 1 | `create_vertex` | CV | ✅ param | V+1, E+0, F+1 |
| 2 | `delete_vertex` | — | — | V−1, E+0, F−1 |
| 3 | `insert_edge` | IE | ✅ identity | E+1, ±F |
| 4 | `delete_edge` | DE | ✅ identity | E−1, ±F |
| 5 | `extrude_face` | — | ✅ torch | V+n, E+2n, F+n |
| 6 | `stellate` | — | ✅ torch | V+1, E+n, F+n−1 |
| 7 | `subdivide_edge` | — | ✅ torch | V+1, E+1, F+0 |
| 8 | `subdivide_face` | — | ✅ torch | V+1, E+n, F+n−1 |
| 9 | `add_handle` | HDL | ✅ identity | χ−2, genus+1 |
| 10 | `stellate_all` | STA | ✅ linear | V+F, 3E, 2E |
| 11 | `catmull_clark` | CC | ✅ linear | V+E+F, 4E, 2E |
| 12 | `dual` | DUAL | ✅ linear | F, E, V |
| 13 | `doo_sabin` | DS | ✅ linear | 2E, 4E, V+E+F |
| 14 | `simplest_subdivide` | SIMP | ✅ linear | E, 2E, F+V |
| 15 | `vertex_cutting` | VC | ✅ linear | 2E, 3E, F+V |
| 16 | `loop_subdivide` | LOOP | ✅ linear | V+E, 4E, 4F |
| 17 | `sqrt3_subdivide` | SQRT3 | ✅ linear | V+F, 3E, 3F |
| 18 | `honeycomb_subdivide` | HONEY | ✅ linear | 2E, 3E, F+V |
| 19 | `star_subdivide` | STAR | ✅ torch | V+F+2E, 9E, 6E |
| 20 | `corner_cutting` | CCUT | ✅ linear | 2E, 4E, V+E+F |
| 21 | `loop_style_subdivide` | LSTYLE | ✅ linear | V+E, 4E, F+2E |
| 22 | `fractal_subdivide` | FRAC | ✅ torch | V+E+F, 6E, 4E |
| 23 | `pentagonal_subdivide` | PENT | ✅ linear | V+2E+F, 5E, 2E |
| 24 | `pentagonal2_subdivide` | PENT2 | ✅ linear | V+3E, 6E, F+2E |
| 25 | `dual1264_subdivide` | D1264 | ✅ linear | 4E, 6E, F+E+V |
| 26 | `root4_subdivide` | ROOT4 | ✅ linear | V+2E, 4E, F+E |
| 27 | `checkerboard_remesh` | CHKB | ✅ linear | V+4E, 9E, F+4E |
| 28 | `ds_bc_new_subdivide` | DSBC | ✅ linear | V+4E, 7E, F+2E |
| 29 | `dome_subdivide` | DOME | ✅ torch | V+59E, 116E, F+56E |
| — | `create_crust` | CRUST | ✅ torch | 2V, 2E, 2F |

**✅ linear** = sparse matrix trace (gradients to vertex positions)
**✅ torch** = dedicated torch implementation (gradients to positions + parameters)

## Differentiable Geometry

```python
import torch
from topmod.diffgeo import DiffSequence

# Compose operators into an end-to-end differentiable map
seq = DiffSequence("cube").append("DS").append("CC").append("CRUST", thickness=0.1)
final_verts = seq.forward()     # differentiable w.r.t. seq.verts0 (8 cube vertices)
tris = seq.triangles()          # int64 [T, 3] for nvdiffrast

loss = (final_verts ** 2).sum()
loss.backward()                 # gradients reach the 8 base-primitive vertices
```

See [`docs/diffgeo.md`](docs/diffgeo.md) for the full API.

## Project Structure

```
topmod/              Pure-Python DLFL mesh library (zero dependencies)
  dlfl.py            Half-edge data structure
  operators.py       4 fundamental operators (Akleman & Chen 2003)
  high_level_ops.py  Extrude, stellate, add_handle, subdivide
  subdivision.py     Catmull-Clark
  remeshing.py       20 remeshing schemes (clean-room from reference semantics)
  diffgeo.py         PyTorch differentiable geometry (optional torch dependency)
  tokenizer.py       Autoregressive mesh tokenizer
  validate.py        2-manifold validation
  io.py              OBJ import/export

blender_addon/       Blender addon (3.6+)
  topmod_blender/    Installable addon package
  topmod_blender.zip Pre-built zip for Install from Disk
  build_zip.sh       Rebuild script
  README.md          Installation & usage guide

pipeline/            Differentiable rendering pipeline
  geometry_optimizer.py   nvdiffrast silhouette fitting
  topology_builder.py     DLFL → GPU tensor export
  manifold_loss.py        Differentiable manifold loss

docs/
  operators.md       Full operator reference (#1–#29) with visualizations
  diffgeo.md         Differentiable geometry API
  reference_semantics.md  Clean-room semantics extraction
  vocabulary_roadmap.md   Token vocabulary evolution

tests/               465 tests (381 core + 84 diffgeo)
scripts/             Gallery generator
```

## References

- Akleman & Chen 2003 — "A minimal and complete set of operators for the
  development of robust manifold mesh modelers"

---
*Zengyn42 · Genesis Research*
