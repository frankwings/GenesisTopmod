# TopMod for Blender — Installation & Usage Guide

A Blender addon bringing all 21 global TopMod mesh operators into
Blender's Edit Mode, based on Akleman & Chen's DLFL (Doubly-Linked
Face List) theory. Every operation preserves 2-manifoldness at every step.

## Requirements

- **Blender 3.6+** (tested on 4.5.7 LTS)
- The mesh must be a **closed, orientable 2-manifold** (no loose vertices,
  no boundary edges, no non-manifold edges). Blender's default cube,
  UV sphere, ico sphere, and any watertight mesh work.

## Installation

### Method 1: Install from Disk (recommended)

1. Download or build `topmod_blender.zip` (run `bash blender_addon/build_zip.sh`)
2. Open Blender
3. Go to **Edit → Preferences → Add-ons**
4. Click **Install...** (top right)
5. Navigate to `topmod_blender.zip` and click **Install Add-on**
6. Enable the addon by checking the box next to **"Mesh: TopMod (DLFL Mesh Operators)"**

### Method 2: Manual copy

Copy the `topmod_blender/` folder to your Blender addons directory:

| OS | Path |
|---|---|
| Windows | `%APPDATA%\Blender Foundation\Blender\4.5\scripts\addons\` |
| Linux | `~/.config/blender/4.5/scripts/addons/` |
| macOS | `~/Library/Application Support/Blender/4.5/scripts/addons/` |

Then enable in Preferences → Add-ons.

## Usage

### Accessing the operators

All operators require **Edit Mode** on a mesh object:

1. Select a mesh object
2. Press **Tab** to enter Edit Mode
3. Access operators via either:
   - **Menu**: Mesh → TopMod → [choose operator]
   - **Sidebar**: Press **N** to open the sidebar → **TopMod** tab
   - **Search**: Press **F3** and type the operator name (e.g. "Catmull")

### Available operators (21)

#### High-Level
| Operator | Description |
|---|---|
| **Stellate All** | Pyramid on every face → all-triangle mesh |

#### Classic Subdivision
| Operator | Parameters | Description |
|---|---|---|
| **Catmull-Clark** | — | Industry-standard smooth subdivision (all-quad output) |
| **Dual** | — | Combinatorial dual: faces ↔ vertices |
| **Doo-Sabin** | — | Corner-cutting subdivision |
| **Simplest** | — | Mid-edge / Peters-Reif subdivision |
| **Vertex Cutting** | offset (0.01–0.49) | Truncate every vertex |
| **Loop** | — | Loop subdivision (triangle meshes only) |
| **√3** | — | Kobbelt √3 subdivision (triangle meshes only) |

#### TopMod Remeshing
| Operator | Parameters | Description |
|---|---|---|
| **Honeycomb** | — | Dual of stellate-all → hexagon-dominated mesh |
| **Star** | offset (0–2) | Stellate-all ×2 with optional spike height |
| **Corner Cutting** | alpha (0.01–0.99) | Parameterized Doo-Sabin variant |
| **Loop-Style** | length (0–1) | Loop connectivity for arbitrary polygons |
| **Fractal** | offset (0–5) | Loop-style + stellated spikes |
| **Pentagonal** | offset (0–1) | All-pentagon output |
| **Pentagonal 2** | scale (0.1–1) | Inner d-gon + surrounding pentagons |
| **Dual 12.6.4** | scale (0.1–2) | Dodecagon/hexagon/quad tiling |
| **Root-4** | smoothing, twist | Honeycomb-mask inner polygons |
| **Checkerboard** | thickness (0.01–0.49) | Alternating quad pattern |
| **DS BC-New** | scale, length | Doo-Sabin variant with surviving vertices |
| **Dome** | length, scale | 7-layer extrusion domes on every face |

#### Structural
| Operator | Parameters | Description |
|---|---|---|
| **Create Crust** | thickness (−2 to 2) | Hollow shell (duplicate + inward offset) |

### Workflow example

1. Add a cube (**Shift+A → Mesh → Cube**)
2. Enter Edit Mode (**Tab**)
3. Open the TopMod sidebar panel (**N** → TopMod tab)
4. Click **Catmull-Clark** — the cube becomes a smooth 26-vertex all-quad mesh
5. Click **Honeycomb** — the quads become hexagons
6. Adjust parameters in the operator popup (bottom-left of viewport) or
   press **F9** to reopen it
7. **Ctrl+Z** to undo any step (full undo support)

### Tips

- **Loop** and **√3** only work on pure triangle meshes — apply
  **Stellate All** first to triangulate any mesh.
- **Dome** generates many vertices (V + 59E) — use on low-poly meshes.
- **Create Crust** produces two disconnected shells; use Blender's
  **Separate by Loose Parts** (P → By Loose Parts) to split them.
- Parameters can be adjusted **after** applying (F9 or bottom-left popup)
  — Blender re-runs the operator with the new values.

## Verification

The addon was tested headlessly on Blender 4.5.7 LTS:

```
CC on cube: V=26 E=48 F=24          ← matches oracle V'=V+E+F, F'=2E
DS on CC:   V=96 E=192 F=98         ← matches oracle V'=2E, F'=V+E+F
Honeycomb:  V=24 E=36 F=14          ← matches oracle V'=2E, F'=F+V
Crust:      V=16 E=24 F=12          ← matches oracle V'=2V, F'=2F
Star:       V=38 E=108 F=72         ← matches oracle V'=V+F+2E, F'=6E
Pentagonal: V=38 E=60 F=24          ← matches oracle V'=V+2E+F, F'=2E
```

All element counts match the closed-form oracles in `docs/operators.md`.

## Architecture

```
topmod_blender/
  __init__.py       ← bl_info + register/unregister
  converter.py      ← BMesh ↔ DLFLMesh bidirectional conversion
  operators.py      ← 21 bpy.types.Operator classes (factory pattern)
  panels.py         ← Mesh menu + N-panel sidebar
  topmod/           ← Bundled pure-Python topmod core (zero dependencies)
```

The topmod core has **zero external dependencies** — no NumPy, no PyTorch.
It is bundled directly inside the addon as a sub-package. The `converter.py`
module is the sole interface between Blender's BMesh and the DLFL mesh
representation.

## For developers

To update the bundled topmod core after modifying `topmod/*.py`:

```bash
bash blender_addon/build_zip.sh
```

This copies the latest core files (excluding torch-dependent `diffgeo.py`
and `tokenizer.py`) and rebuilds the zip.
