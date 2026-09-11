# TopMod for Blender — Installation & Usage Guide

A Blender addon bringing all 46 TopMod mesh operators into
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

### Selection modes

Different operators require different selection modes:

| Selection Mode | How to activate | Operators |
|---|---|---|
| **No selection needed** | — | All global/subdivision/remeshing operators |
| **Face selection** | Press **3** | Extrude Face, Stellate, Subdivide Face, Triangulate Face, Double Stellate Face, Extrude Face Dome |
| **Edge selection** | Press **2** | Subdivide Edge, Collapse Edge, Trisect Edge |
| **Two-face selection** | Press **3**, select 2 faces | Add Handle, Punch Hole |
| **Vertex selection** | Press **1**, click 1 vertex | Delete Vertex |
| **Interactive picking** | — run the operator, then click in the viewport | Insert Edge (2 corners) |

### Insert Edge — picking the two corners

`insert_edge` joins two **corners**. A corner is a (face, vertex) pair — not
just a vertex — because a vertex shared by several faces offers a corner on
each of them, and which one you mean decides what happens.

Insert Edge picks them for you interactively, so nothing has to be selected
beforehand:

1. Run **Mesh → TopMod → Edge Ops → Insert Edge** (or the sidebar button)
2. **Click a face.** It highlights orange under the cursor, blue once chosen
3. **Click one of its corners.** Corners show as white dots; the one under
   the cursor gets a yellow wedge along its two face edges
4. **Click the second corner** on the same face. A preview line follows the
   cursor from the first corner, and the first corner stays green

The viewport header shows the current step throughout.

| Key | Action |
|---|---|
| **Left click** | Take the highlighted face / corner |
| **Backspace** | Step back exactly one step |
| **Esc** or **right click** | Cancel; the original selection and select mode come back |
| **MMB / wheel / numpad / NDOF** | Orbit, pan and zoom as usual, mid-pick |

The new edge is selected when the operator finishes, and the whole
interaction plus the insertion is a single undo step. The **Redo panel**
re-runs it from the recorded face and corner indices.

**Same face or two faces.** Both corners on one face **splits** it in two.
Corners on two *different* faces **merges** them into a single face whose
boundary runs through the new edge once per direction, which opens a handle
(genus +1) — the `insert_edge_cross` case in
[docs/operators.md](../docs/operators.md). Both are picked identically.

**Which corners are refused.** Almost nothing. A corner turns red, with the
reason in the header, in only two cases — and both are because the result
*hangs*, not because the topology is wrong:

| Refused pick | Why |
|---|---|
| The very same corner twice | `insert_edge` splices both new half-edges against one half-edge, leaving a face loop that never closes; the next traversal spins forever |
| Both corners on the same vertex | The self-loop edge that makes is stored by Blender perfectly well, but `bm.normal_update()` and clearing the selection then spin forever on it. TopMod proper allows this; the Blender bridge cannot |

Everything else goes through — including the two cases stock Blender
modelling cannot express:

- **Corners on two different faces** merge the faces into one whose boundary
  runs through the new edge once per direction (genus +1).
- **Corners whose vertices are already joined** add a *second, parallel*
  edge beside the existing one, bounding a 2-gon. Neighbouring corners of
  one face are the everyday case.

### How the exotic topology is stored

DLFL meshes hold three things a Blender mesh cannot be *built* with:

| Shape | Produced by | What refuses to build it |
|---|---|---|
| A face that visits a vertex twice | cross-face insert | `bmesh.faces.new()` — "found the same (BMVert) used multiple times" |
| Two edges on one vertex pair | inserting between joined corners | `bmesh.edges.new()` — "this edge exists" |
| A 2-gon | the same | nothing; it is fine once built |

Blender *stores* all of them — verified on 4.4 and 5.3 through Edit Mode
round trips and `.blend` save/reload. Only the constructors refuse. So
`converter._load_dlfl_into_bmesh` fills a scratch mesh's arrays directly,
setting each loop's **`edge_index`** as well as its vertex, and hands that
to `bm.from_mesh`. Writing the edge per corner is what keeps two parallel
edges apart; `Mesh.from_pydata` would look each corner's edge up by vertex
pair and silently merge them.

Reading back is the same story in reverse: `bmesh_to_dlfl` takes each face
as its loops' `(vertex index, edge index)` pairs, and
`corner_rules.build_dlfl_from_corners` pairs half-edges by **edge index**.
The core's `primitives._build_mesh` keys them by directed vertex pair, so
two parallel edges would overwrite each other and one would lose its twin —
which is why the converter no longer uses it. A side benefit: `_be_map` is
now keyed on Blender's own edge indices, so the selected-edge operators
target the right one of a parallel pair.

Three consequences worth knowing:

- **Never call `mesh.validate()`** on a mesh in this state — it deletes
  duplicate edges *and* faces that repeat a vertex. The addon does not, but
  a stock Blender operator that validates will clean the topology up.
- **A one-corner polygon crashes Blender** outright on Mesh→BMesh
  conversion. `insert_edge` cannot make one; `create_vertex` can, as its
  degenerate loop face, so the writer drops those.
- **The subdivision operators do not handle merged faces yet.** Running
  Catmull-Clark on one yields a non-manifold result. That is a limitation of
  the topmod core, not the Blender bridge — it reproduces in pure Python.
  Insert Edge itself chains fine.

### Available operators (46)

#### Fundamental (Akleman & Chen 2003)
| Operator | Selection | Description |
|---|---|---|
| **Insert Edge** | 4 vertices (ordered) | Insert edge between two half-edges |
| **Delete Edge** | 1 edge | Delete an edge, merging flanking faces |

#### High-Level (selection-based)
| Operator | Selection | Parameters | Description |
|---|---|---|---|
| **Extrude Face** | 1+ faces | dist (0–5) | Pull face outward along normal |
| **Stellate** | 1+ faces | dist (0–5) | Pyramid apex on face |
| **Subdivide Edge** | 1+ edges | — | Split edge at midpoint |
| **Subdivide Face** | 1+ faces | — | Fan from centroid |
| **Collapse Edge** | 1+ edges | — | Merge endpoints to midpoint |
| **Trisect Edge** | 1+ edges | — | Split edge into 3 segments |
| **Triangulate Face** | 1+ faces | — | Fan triangulation |
| **Double Stellate Face** | 1+ faces | dist (0–5) | Two-level stellate spike |
| **Extrude Face Dome** | 1+ faces | length, scale | 5-layer dome on face |
| **Add Handle** | 2 faces | — | Tunnel between two faces (genus +1) |
| **Punch Hole** | 2 faces | — | Alias for Add Handle |
| **Delete Vertex** | 1 vertex | — | Remove isolated vertex |

#### Global (no selection needed)
| Operator | Description |
|---|---|
| **Stellate All** | Pyramid on every face → all-triangle |
| **Subdivide All Edges** | Midpoint split every edge |
| **Subdivide All Faces** | Centroid fan every face |
| **Triangulate All** | Triangulate every face |
| **Stellate Subdivide** | Stellate all + delete original edges |
| **Make Wireframe** | MCC → crust → punch holes → hollow beams |

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
| **Doo-Sabin BC** | — | Subdivide all edges then Doo-Sabin |
| **Two-Stellate** | offset, curve | Two-pass stellate subdivision |
| **Modified Corner Cutting** | thickness | Bisector-based inset + bridge |
| **Modified Corner Cutting 2** | scale | Uniform displacement variant |

#### Structural
| Operator | Parameters | Description |
|---|---|---|
| **Create Crust** | thickness (−2 to 2) | Hollow shell (duplicate + normal offset) |
| **Create Crust (Scaling)** | scale (0.1–1) | Hollow shell (scale toward centroid) |

### Workflow examples

#### Basic subdivision

1. Add a cube (**Shift+A → Mesh → Cube**)
2. Enter Edit Mode (**Tab**)
3. Open the TopMod sidebar panel (**N** → TopMod tab)
4. Click **Catmull-Clark** — the cube becomes a smooth 26-vertex all-quad mesh
5. Click **Honeycomb** — the quads become hexagons
6. Adjust parameters in the operator popup (bottom-left of viewport) or
   press **F9** to reopen it
7. **Ctrl+Z** to undo any step (full undo support)

#### Face extrusion

1. Start with a cube in Edit Mode
2. Switch to **Face selection** (press **3**)
3. Select one face
4. Click **Extrude Face** in the sidebar → a box grows from that face
5. Select the new top face, click **Stellate** → a pyramid forms

#### Insert Edge (same face — diagonal)

1. Start with a cube in Edit Mode
2. Switch to **Vertex selection** (press **1**)
3. Click vertex A (one corner of the top face)
4. **Shift+click** vertex B (the adjacent corner — this defines half-edge 1: A→B)
5. **Shift+click** vertex C (the opposite corner of the same face)
6. **Shift+click** vertex D (the remaining corner — this defines half-edge 2: C→D)
7. Run **Insert Edge** → a diagonal splits the top face into two triangles

#### Wireframe generation

1. Start with any mesh in Edit Mode
2. Click **Make Wireframe** (thickness=0.15)
3. The solid becomes a hollow wireframe with beams along every edge

### Tips

- **Loop** and **√3** only work on pure triangle meshes — apply
  **Stellate All** or **Triangulate All** first.
- **Dome** generates many vertices (V + 59E) — use on low-poly meshes.
- **Create Crust** produces two disconnected shells; use Blender's
  **Separate by Loose Parts** (P → By Loose Parts) to split them.
- Parameters can be adjusted **after** applying (F9 or bottom-left popup)
  — Blender re-runs the operator with the new values.
- For **Insert Edge**, always click vertices one by one (Shift+click).
  Box select or Select All loses the click order and will fail.

## Verification

The addon was tested on Blender 4.5.7 LTS (46 operators, all passing):

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
  converter.py      ← BMesh ↔ DLFLMesh conversion + selection helpers
  operators.py      ← 46 bpy.types.Operator classes (factory pattern)
  corner_pick.py    ← Insert Edge: modal corner picker + viewport overlays
  corner_rules.py   ← Corner rules + the lossless DLFL <-> corner-array form
  panels.py         ← Mesh menu + N-panel sidebar
  topmod/           ← Bundled pure-Python topmod core (zero dependencies)
```

The topmod core has **zero external dependencies** — no NumPy, no PyTorch.
It is bundled directly inside the addon as a sub-package. The `converter.py`
module is the sole interface between Blender's BMesh and the DLFL mesh
representation.

Selection-based operators use helper functions in `converter.py`:
- `apply_local_face_op()` — operates on selected faces
- `apply_local_edge_op()` — operates on selected edges
- `apply_two_face_op()` — requires exactly 2 selected faces
- `apply_insert_edge_corners()` — takes two (face index, corner position) pairs from the picker
- `apply_delete_vertex()` — operates on 1 selected vertex

## For developers

To update the bundled topmod core after modifying `topmod/*.py`:

```bash
bash blender_addon/build_zip.sh
```

This copies the latest core files (excluding torch-dependent `diffgeo.py`
and `tokenizer.py`) and rebuilds the zip.

### Tests

The corner rules are pure Python and run under pytest with the rest of the
suite — `corner_rules.py` imports no `bpy` on purpose:

```bash
python -m pytest tests/test_corner_rules.py
```

Everything reachable through `execute()` — registration, the corner →
half-edge adapter, the refusals, the resulting selection — runs headlessly:

```bash
blender -b --factory-startup --python blender_addon/test_headless.py
```

The modal picker itself needs a real 3D Viewport. Check it by hand:

| Scenario | Expected |
|---|---|
| Two non-adjacent corners of one cube face | The face splits; one undo reverts everything |
| One corner on each of two different faces | The two faces merge; the mesh stays closed and genus goes up by 1 |
| Two neighbouring corners of one face | A second edge appears beside the one they share, bounding a 2-gon; the mesh stays closed |
| Hover the stored corner itself, or any corner on its vertex | Red wedge, red preview line, reason in the header, click ignored |
| Backspace at each step | Steps back exactly one step |
| Esc / right click at each step | Original selection and select mode restored; overlays and header text gone |
| Orbit, pan, zoom mid-pick (MMB, wheel, trackpad, numpad, NDOF) | Navigation works; hover updates on the next mouse move |
| Object with rotation and non-uniform scale | Face and corner picking land on the right elements |
| Perspective, orthographic, quad view, X-ray on | Picking and overlays correct; labels in the right place |
| UI scale 2.0 | Overlay sizes and hover radius scale with it |
| Mesh with hidden faces | Hidden faces cannot be picked |
| Adjust the Redo panel after confirming | Undo and re-execute with the new values works |
| Leave Edit Mode mid-pick | Operator cancels; no draw errors in the console afterwards |
