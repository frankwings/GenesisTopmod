#!/usr/bin/env python3
"""
Generate before/after visualizations for every TopMod operator, plus the
docs/operators.md reference that links them.

Usage:
    python3 scripts/generate_op_gallery.py            # images + markdown
    python3 scripts/generate_op_gallery.py --md-only  # regenerate markdown only

Output:
    docs/assets/ops/<name>.png   — side-by-side before/after render
    docs/operators.md            — full operator reference (generated)

The registry below is the single source of truth: adding an operator here
updates both the gallery and the markdown.
"""

from __future__ import annotations

import os
import sys
import argparse
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from topmod import (
    DLFLMesh,
    make_cube, make_tetrahedron, make_icosahedron,
    insert_edge, delete_edge,
    extrude_face, add_handle, stellate, stellate_all,
    subdivide_edge, subdivide_face,
    catmull_clark,
    dual, doo_sabin, simplest_subdivide, vertex_cutting,
    loop_subdivide, sqrt3_subdivide,
    honeycomb_subdivide, star_subdivide, corner_cutting,
    loop_style_subdivide, fractal_subdivide,
    pentagonal_subdivide, pentagonal2_subdivide,
    dual1264_subdivide, root4_subdivide,
    checkerboard_remesh, ds_bc_new_subdivide, dome_subdivide,
    create_crust,
)

ASSET_DIR = os.path.join(ROOT, "docs", "assets", "ops")
MD_PATH   = os.path.join(ROOT, "docs", "operators.md")


# ─────────────────────────────────────────────────────────────────────────────
# Rendering
# ─────────────────────────────────────────────────────────────────────────────

def _render(ax, mesh: DLFLMesh, alpha: float = 0.92) -> None:
    polys = []
    for f in mesh.iter_faces():
        polys.append([(v.x, v.y, v.z) for v in f.vertices()])
    coll = Poly3DCollection(polys, alpha=alpha, linewidths=0.6)
    coll.set_facecolor("#a8c4e0")
    coll.set_edgecolor("#1a2a3a")
    ax.add_collection3d(coll)

    xs = [v.x for v in mesh.iter_vertices()]
    ys = [v.y for v in mesh.iter_vertices()]
    zs = [v.z for v in mesh.iter_vertices()]
    lo = min(min(xs), min(ys), min(zs))
    hi = max(max(xs), max(ys), max(zs))
    pad = 0.05 * (hi - lo + 1e-9)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_zlim(lo - pad, hi + pad)
    ax.set_box_aspect((1, 1, 1))
    ax.set_axis_off()


def render_pair(name: str, before: DLFLMesh, after: DLFLMesh,
                alpha_after: float = 0.92) -> str:
    fig = plt.figure(figsize=(8, 4.2))
    for i, (mesh, title, alpha) in enumerate(
            [(before, "before", 0.92), (after, "after", alpha_after)]):
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        _render(ax, mesh, alpha=alpha)
        ax.set_title(f"{title}  V={mesh.V()} E={mesh.E()} F={mesh.F()}",
                     fontsize=9)
    fig.suptitle(name, fontsize=12)
    fig.tight_layout()
    out = os.path.join(ASSET_DIR, f"{name}.png")
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Operator registry — single source of truth for gallery + markdown
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class OpEntry:
    name: str                     # image/file name
    category: str
    signature: str
    token: str                    # tokenizer opcode or '—'
    oracle: str                   # closed-form element-count effect
    params: str                   # parameter meaning ('—' if none)
    desc: str                     # short explanation of what it does
    example: str                  # usage snippet
    base: Callable[[], DLFLMesh] = make_cube
    base_name: str = "cube"
    # apply(mesh) -> mesh_after (may mutate in place and return same mesh)
    apply: Optional[Callable[[DLFLMesh], DLFLMesh]] = None
    alpha_after: float = 0.92
    no_image: bool = False
    no_image_reason: str = ""
    # differentiability status key (see _DIFF table below)
    diff: str = "pending"


# Differentiability display strings (torch support via topmod/diffgeo.py).
# Topology is always discrete; "differentiable" refers to the position map
# (output vertex coordinates as a function of input coordinates) with the
# operator sequence held fixed.
_DIFF = {
    "linear":   ("✅ Yes", "linear — traced to a sparse matrix in "
                           "`topmod/diffgeo.py`; gradients flow to input "
                           "vertex positions (op parameters baked as "
                           "constants)"),
    "identity": ("✅ Yes", "pure topology, no coordinates created — the "
                           "position map is the identity"),
    "param":    ("✅ Yes", "the position is itself a free parameter (leaf "
                           "tensor)"),
    "crust":    ("✅ Yes", "dedicated torch implementation in "
                           "`topmod/diffgeo.py`; gradients flow to input "
                           "positions **and** to `thickness`"),
    "pending":  ("⏳ Not yet", "smooth almost everywhere (face normals / "
                           "edge lengths) but the torch implementation is "
                           "phase 2 — not in `topmod/diffgeo.py` yet"),
    "local":    ("⏳ Not yet", "linear in principle, but local single-element "
                           "operators are not yet wired into the "
                           "`topmod/diffgeo.py` tracer (phase 2)"),
    "none":     ("—", "no geometry to differentiate"),
}


def _first_face(m): return next(iter(m.faces.values()))
def _first_edge(m): return next(iter(m.edges.values()))


def _apply_insert_edge(m):
    f = _first_face(m)
    hes = f.halfedges()
    insert_edge(m, hes[0], hes[2])   # diagonal chord across the quad
    return m


def _apply_delete_edge(m):
    delete_edge(m, _first_edge(m))   # merge the two flanking faces
    return m


def _apply_extrude(m):
    extrude_face(m, _first_face(m), dist=0.6)
    return m


def _apply_stellate(m):
    stellate(m, _first_face(m))
    return m


def _apply_subdivide_edge(m):
    subdivide_edge(m, _first_edge(m))
    return m


def _apply_subdivide_face(m):
    subdivide_face(m, _first_face(m))
    return m


def _apply_add_handle(m):
    faces = list(m.faces.values())
    add_handle(m, faces[0], faces[1])    # tunnel between two OPPOSITE faces
    return m


def _apply_stellate_all(m):
    stellate_all(m)
    return m


def _apply_star(m):
    star_subdivide(m, offset=0.15)
    return m


def _apply_dome(m):
    dome_subdivide(m)
    return m


def _apply_crust_hole(m):
    out, pairs = create_crust(m, thickness=0.25)
    add_handle(out, pairs[0][0], pairs[0][1])   # punch one hole to reveal shell
    return out


OPS: List[OpEntry] = [
    # ── 1. Fundamental operators (Akleman & Chen 2003 minimal complete set) ──
    OpEntry("create_vertex", "1. Fundamental Operators",
            "create_vertex(mesh, x, y, z) -> Vertex", "CV",
            "V+1, E+0, F+1 (point sphere)",
            "x, y, z — coordinates",
            "Creates an isolated *point sphere*: a single vertex that forms "
            "its own connected component with one degenerate face. It is the "
            "starting point of every DLFL construction — all meshes grow from "
            "point spheres by inserting edges.",
            "v = create_vertex(mesh, 0.0, 0.0, 0.0)",
            no_image=True,
            no_image_reason="a single point — nothing meaningful to render"),
    OpEntry("delete_vertex", "1. Fundamental Operators",
            "delete_vertex(mesh, vertex)", "—",
            "V−1, E+0, F−1",
            "—",
            "Removes an isolated point sphere. Only legal on a vertex with no "
            "incident edges; it is the exact inverse of `create_vertex`.",
            "delete_vertex(mesh, v)",
            no_image=True,
            no_image_reason="a single point — nothing meaningful to render"),
    OpEntry("insert_edge", "1. Fundamental Operators",
            "insert_edge(mesh, he1, he2) -> Edge", "IE",
            "E+1; same face → F+1 (split), different faces → F−1 "
            "(merge components / open handle)",
            "he1, he2 — two corners (half-edges)",
            "Inserts a new edge between two corners. If both corners lie on "
            "the *same* face, the face is split in two (shown: a diagonal "
            "chord splits a cube quad into two triangles). If they lie on "
            "*different* faces, the two faces merge into one — this is how "
            "components are joined and handles are opened. One of the two "
            "core DLFL operators; the mesh is a valid 2-manifold after every "
            "single call.",
            "hes = face.halfedges()\n"
            "insert_edge(mesh, hes[0], hes[2])  # diagonal across the quad",
            apply=_apply_insert_edge),
    OpEntry("delete_edge", "1. Fundamental Operators",
            "delete_edge(mesh, edge)", "DE",
            "E−1; two distinct sides → F−1 (merge), same face both sides → F+1",
            "—",
            "Deletes an edge. When the two sides of the edge belong to "
            "different faces, those faces merge into one (shown: removing one "
            "cube edge merges two squares into a hexagon). The inverse of "
            "`insert_edge`.",
            "delete_edge(mesh, edge)",
            apply=_apply_delete_edge),

    # ── 2. High-level operators ─────────────────────────────────────────────
    OpEntry("extrude_face", "2. High-Level Operators",
            "extrude_face(mesh, face, dist=1.0) -> List[Face]", "—",
            "V+n, E+2n, F+n (n = face degree)",
            "dist — extrusion distance along the face normal",
            "Extrudes a face along its normal, creating a lifted copy of the "
            "face (the *top*) connected to the original boundary by n side "
            "quads — like pulling a box out of the surface. Returns "
            "`[top_face] + side_faces`. The building block of the DOME scheme.",
            "new_faces = extrude_face(mesh, face, dist=0.6)\n"
            "top = new_faces[0]",
            apply=_apply_extrude),
    OpEntry("stellate", "2. High-Level Operators",
            "stellate(mesh, face, dist=0.0) -> Vertex", "—",
            "V+1, E+n, F+n−1",
            "dist — apex displacement along the face normal",
            "Stellates one face: adds an apex vertex at the face centroid "
            "(optionally raised along the normal) and connects it to every "
            "corner, turning one n-gon into n triangles — a pyramid grown on "
            "the face.",
            "apex = stellate(mesh, face)",
            apply=_apply_stellate),
    OpEntry("subdivide_edge", "2. High-Level Operators",
            "subdivide_edge(mesh, edge) -> Vertex", "—",
            "V+1, E+1, F+0",
            "—",
            "Splits an edge at its midpoint and returns the new midpoint "
            "vertex. The two flanking faces each gain one corner; no face is "
            "created or destroyed.",
            "mid = subdivide_edge(mesh, edge)",
            apply=_apply_subdivide_edge),
    OpEntry("subdivide_face", "2. High-Level Operators",
            "subdivide_face(mesh, face) -> Vertex", "—",
            "V+1, E+n, F+n−1",
            "—",
            "Fans a face from its centroid: a center vertex is added and "
            "connected to every corner. Topologically identical to "
            "`stellate`, but the new vertex stays in the face plane instead "
            "of being lifted.",
            "c = subdivide_face(mesh, face)",
            apply=_apply_subdivide_face),
    OpEntry("add_handle", "2. High-Level Operators",
            "add_handle(mesh, face1, face2) -> List[Edge]", "HDL",
            "V+0, E+n, F+n−2, χ−2, genus+1 (same component)",
            "—",
            "Connects two faces of equal degree with a tunnel (handle): both "
            "faces are consumed and n side quads bridge their boundaries. "
            "This is the only operator that changes genus — a cube becomes a "
            "square torus (shown: tunnel between two opposite faces). It is "
            "also the hole-punching primitive for `create_crust` shells.",
            "add_handle(mesh, top_face, bottom_face)",
            apply=_apply_add_handle, alpha_after=0.55),
    OpEntry("stellate_all", "2. High-Level Operators",
            "stellate_all(mesh) -> List[Vertex]  # in place", "STA",
            "V'=V+F, E'=3E, F'=2E",
            "—",
            "Stellates every face of the mesh at once, producing an "
            "all-triangle mesh (a pyramid on every face). Used as a building "
            "block inside the honeycomb and star schemes.",
            "apexes = stellate_all(mesh)",
            apply=_apply_stellate_all),

    # ── 3. Classic subdivision ──────────────────────────────────────────────
    OpEntry("catmull_clark", "3. Classic Subdivision",
            "catmull_clark(mesh) -> DLFLMesh", "CC",
            "V'=V+E+F, E'=4E, F'=2E (all quads)",
            "—",
            "Catmull-Clark subdivision, the industry-standard smoothing "
            "scheme: face points, edge points and repositioned vertex points "
            "split every face into quads while pulling the surface toward a "
            "smooth limit surface. Output is always an all-quad mesh.",
            "out = catmull_clark(mesh)",
            apply=catmull_clark),
    OpEntry("dual", "3. Classic Subdivision",
            "dual(mesh) -> DLFLMesh", "DUAL",
            "V'=F, E'=E, F'=V",
            "—",
            "Takes the combinatorial dual: every face becomes a vertex (at "
            "its centroid) and every vertex becomes a face. Applying it twice "
            "returns the original topology: `dual(dual(M)) ≅ M`. A cube maps "
            "to an octahedron and vice versa.",
            "out = dual(mesh)",
            apply=dual),
    OpEntry("doo_sabin", "3. Classic Subdivision",
            "doo_sabin(mesh) -> DLFLMesh", "DS",
            "V'=2E, E'=4E, F'=V+E+F",
            "—",
            "Doo-Sabin subdivision (corner-cutting family): one new vertex "
            "per face corner, producing shrunken *face-faces*, quad "
            "*edge-faces*, and *vertex-faces* — every sharp corner and edge "
            "of the input gets beveled away.",
            "out = doo_sabin(mesh)",
            apply=doo_sabin),
    OpEntry("simplest_subdivide", "3. Classic Subdivision",
            "simplest_subdivide(mesh) -> DLFLMesh", "SIMP",
            "V'=E, E'=2E, F'=F+V",
            "—",
            "Mid-edge (simplest / Peters-Reif) subdivision: edge midpoints "
            "become the only vertices; each face shrinks to its midpoint "
            "polygon and each old vertex is replaced by a new face. A cube "
            "becomes a cuboctahedron.",
            "out = simplest_subdivide(mesh)",
            apply=simplest_subdivide),
    OpEntry("vertex_cutting", "3. Classic Subdivision",
            "vertex_cutting(mesh, offset=0.25) -> DLFLMesh", "VC",
            "V'=2E, E'=3E, F'=F+V",
            "offset ∈ (0, 0.5) — corner-cut depth",
            "Vertex truncation: every vertex is sliced off, leaving a small "
            "polygon where the corner was, and every n-gon becomes a 2n-gon. "
            "A cube becomes a truncated cube.",
            "out = vertex_cutting(mesh, offset=0.25)",
            apply=vertex_cutting),
    OpEntry("loop_subdivide", "3. Classic Subdivision",
            "loop_subdivide(mesh) -> DLFLMesh  # triangle meshes only", "LOOP",
            "V'=V+E, E'=4E, F'=4F",
            "—",
            "Loop subdivision: every triangle is split 1-into-4 at edge "
            "midpoints, with β-weighted smoothing of old vertices — the "
            "standard smooth scheme for triangle meshes. Raises ValueError on "
            "non-triangle input.",
            "out = loop_subdivide(make_icosahedron())",
            base=make_icosahedron, base_name="icosahedron",
            apply=loop_subdivide),
    OpEntry("sqrt3_subdivide", "3. Classic Subdivision",
            "sqrt3_subdivide(mesh) -> DLFLMesh  # triangle meshes only", "SQRT3",
            "V'=V+F, E'=3E, F'=3F",
            "—",
            "√3 subdivision (Kobbelt): a vertex is inserted at every face "
            "centroid, then all original edges are flipped, tripling the "
            "triangle count with the slowest possible growth rate. Raises "
            "ValueError on non-triangle input.",
            "out = sqrt3_subdivide(make_icosahedron())",
            base=make_icosahedron, base_name="icosahedron",
            apply=sqrt3_subdivide),

    # ── 4. TopMod remeshing schemes (clean-room, from reference semantics) ──
    OpEntry("honeycomb_subdivide", "4. TopMod Remeshing Schemes",
            "honeycomb_subdivide(mesh) -> DLFLMesh", "HONEY",
            "V'=2E, E'=3E, F'=F+V",
            "—",
            "Honeycomb subdivision, defined as `dual ∘ stellate_all`: "
            "stellate every face, then dualize. Triangle input yields a "
            "hexagon-dominated (honeycomb-like) mesh.",
            "out = honeycomb_subdivide(mesh)",
            apply=honeycomb_subdivide),
    OpEntry("star_subdivide", "4. TopMod Remeshing Schemes",
            "star_subdivide(mesh, offset=0.0)  # in place", "STAR",
            "V'=V+F+2E, E'=9E, F'=6E (all triangles)",
            "offset — first-round apex lift along original face normals",
            "Star subdivision: `stellate_all` applied twice. A positive "
            "offset lifts the first round of apexes along the original face "
            "normals, growing star-like spikes on every face.",
            "star_subdivide(mesh, offset=0.3)",
            apply=_apply_star),
    OpEntry("corner_cutting", "4. TopMod Remeshing Schemes",
            "corner_cutting(mesh, alpha=0.5) -> DLFLMesh", "CCUT",
            "V'=2E, E'=4E, F'=V+E+F (same topology as Doo-Sabin)",
            "alpha ∈ (0, 1) — tension (diagonal weight)",
            "Corner-cutting subdivision: a parameterized geometric variant "
            "of Doo-Sabin with identical connectivity. `alpha` controls how "
            "close each new corner stays to the original corner, i.e. how "
            "aggressively corners are shaved off.",
            "out = corner_cutting(mesh, alpha=0.7)",
            apply=corner_cutting),
    OpEntry("loop_style_subdivide", "4. TopMod Remeshing Schemes",
            "loop_style_subdivide(mesh, length=1.0) -> DLFLMesh", "LSTYLE",
            "V'=V+E, E'=4E, F'=F+2E",
            "length ∈ [0, 1] — old-vertex blend (1 = keep position)",
            "Loop connectivity generalized to arbitrary polygons: each face "
            "gets its corner triangles cut off, leaving a central midpoint "
            "d-gon. On triangle input the connectivity coincides exactly with "
            "Loop subdivision.",
            "out = loop_style_subdivide(mesh)",
            apply=loop_style_subdivide),
    OpEntry("fractal_subdivide", "4. TopMod Remeshing Schemes",
            "fractal_subdivide(mesh, offset=1.0) -> DLFLMesh", "FRAC",
            "V'=V+E+F, E'=6E, F'=4E (all triangles)",
            "offset — spike height factor",
            "Fractal subdivision: `loop_style` followed by stellating every "
            "central polygon with an apex raised along the face normal. "
            "Repeated application produces a fractal, spiky landscape.",
            "out = fractal_subdivide(mesh, offset=1.0)",
            apply=fractal_subdivide),
    OpEntry("pentagonal_subdivide", "4. TopMod Remeshing Schemes",
            "pentagonal_subdivide(mesh, offset=0.0) -> DLFLMesh", "PENT",
            "V'=V+2E+F, E'=5E, F'=2E (all pentagons)",
            "offset ∈ [0, 1] — pull spoke neighbors toward the centroid",
            "Pentagonal subdivision: every edge is trisected and every face "
            "gets a centroid spoke, converting each d-gon into d pentagons — "
            "the whole mesh becomes all-pentagon. A tetrahedron maps to the "
            "combinatorial structure of a regular dodecahedron.",
            "out = pentagonal_subdivide(mesh)",
            apply=pentagonal_subdivide),
    OpEntry("pentagonal2_subdivide", "4. TopMod Remeshing Schemes",
            "pentagonal2_subdivide(mesh, scale_factor=0.75) -> DLFLMesh", "PENT2",
            "V'=V+3E, E'=6E, F'=F+2E",
            "scale_factor — inner-polygon shrink",
            "Second pentagonal variant: edges are split at midpoints and a "
            "scaled inner copy of each face is inserted, then connected — "
            "each face becomes one inner d-gon surrounded by d pentagons.",
            "out = pentagonal2_subdivide(mesh, scale_factor=0.7)",
            apply=pentagonal2_subdivide),
    OpEntry("dual1264_subdivide", "4. TopMod Remeshing Schemes",
            "dual1264_subdivide(mesh, sf=1.0) -> DLFLMesh", "D1264",
            "V'=4E, E'=6E, F'=F+E+V",
            "sf — inner-polygon scale",
            "Dual 12.6.4 subdivision: Doo-Sabin-like, but each face's inner "
            "polygon is a 2d-gon built from the 1/3 and 2/3 points of every "
            "edge. Triangle input produces the semi-regular 12.6.4 tiling "
            "pattern (dodecagons, hexagons, squares).",
            "out = dual1264_subdivide(mesh)",
            apply=dual1264_subdivide),
    OpEntry("root4_subdivide", "4. TopMod Remeshing Schemes",
            "root4_subdivide(mesh, a=0.0, twist=0.0) -> DLFLMesh", "ROOT4",
            "V'=V+2E, E'=4E, F'=F+E",
            "a — old-vertex smoothing blend; twist — inner-ring sampling shift",
            "Root-4 subdivision: an inner polygon (honeycomb-mask weighted) "
            "is inserted in every face and bridged to neighbors with "
            "hexagons, while all original edges are deleted. Unlike "
            "Doo-Sabin, the original vertices survive.",
            "out = root4_subdivide(mesh, a=0.3, twist=0.2)",
            apply=root4_subdivide),
    OpEntry("checkerboard_remesh", "4. TopMod Remeshing Schemes",
            "checkerboard_remesh(mesh, thickness=0.25) -> DLFLMesh", "CHKB",
            "V'=V+4E, E'=9E, F'=F+4E",
            "thickness ∈ (0, 0.5) — inset / trisection ratio",
            "Checkerboard remeshing: each face is inset, each edge trisected, "
            "and corners are chamfered, producing an alternating quad pattern. "
            "Quad input stays all-quad, with a visible checkerboard layout.",
            "out = checkerboard_remesh(mesh, thickness=0.25)",
            apply=checkerboard_remesh),
    OpEntry("ds_bc_new_subdivide", "4. TopMod Remeshing Schemes",
            "ds_bc_new_subdivide(mesh, sf=1.0, length=1.0) -> DLFLMesh", "DSBC",
            "V'=V+4E, E'=7E, F'=F+2E",
            "sf — DS corner scale; length — old-vertex blend",
            "Doo-Sabin \"BC new\" variant: a Doo-Sabin pass is applied to the "
            "midpoint-refined 2d-gon boundary of every face, but the original "
            "vertices survive. Each face yields one 2d-gon plus two pentagons "
            "per edge.",
            "out = ds_bc_new_subdivide(mesh, sf=0.9)",
            apply=ds_bc_new_subdivide),
    OpEntry("dome_subdivide", "4. TopMod Remeshing Schemes",
            "dome_subdivide(mesh, length=1.0, sf=1.0)  # in place", "DOME",
            "V'=V+59E, E'=116E, F'=F+56E",
            "length — height profile scale; sf — ring scale profile",
            "Dome subdivision: every edge is split into quarters, then every "
            "original face is extruded seven times with a built-in "
            "height/scale profile, growing a rounded dome on each face — the "
            "mesh sprouts a bubble on every side.",
            "dome_subdivide(mesh)",
            apply=_apply_dome),

    # ── 5. Structural operators ─────────────────────────────────────────────
    OpEntry("create_crust", "5. Structural Operators",
            "create_crust(mesh, thickness=0.1) -> (DLFLMesh, pairs)", "CRUST",
            "V'=2V, E'=2E, F'=2F, 2 components; after punching k holes: "
            "genus' = 2g+k−1",
            "thickness — shell thickness (negative offsets outward)",
            "Turns a surface into a hollow shell: the whole mesh is duplicated "
            "with reversed orientation and offset inward along averaged vertex "
            "normals, giving an outer and an inner wall. Returns the list of "
            "mirrored face pairs (outer face i ↔ inner face F+i); punching "
            "holes through pairs with `add_handle` connects the walls and "
            "creates tunnels (shown: shell with one hole punched).",
            "out, pairs = create_crust(mesh, thickness=0.25)\n"
            "for outer, inner in pairs[:2]:\n"
            "    add_handle(out, outer, inner)  # each hole: genus +1\n"
            "                                   # (first hole joins the walls)",
            apply=_apply_crust_hole, alpha_after=0.45),
]


# Differentiability status per operator (keys of _DIFF).
_DIFF_BY_NAME = {
    "create_vertex":        "param",
    "delete_vertex":        "none",
    "insert_edge":          "identity",
    "delete_edge":          "identity",
    "extrude_face":         "pending",   # displacement along face normal
    "stellate":             "pending",   # apex displacement along face normal
    "subdivide_edge":       "local",
    "subdivide_face":       "local",
    "add_handle":           "identity",
    "stellate_all":         "linear",
    "catmull_clark":        "linear",
    "dual":                 "linear",
    "doo_sabin":            "linear",
    "simplest_subdivide":   "linear",
    "vertex_cutting":       "linear",
    "loop_subdivide":       "linear",
    "sqrt3_subdivide":      "linear",
    "honeycomb_subdivide":  "linear",
    "star_subdivide":       "pending",
    "corner_cutting":       "linear",
    "loop_style_subdivide": "linear",
    "fractal_subdivide":    "pending",
    "pentagonal_subdivide": "linear",
    "pentagonal2_subdivide": "linear",
    "dual1264_subdivide":   "linear",
    "root4_subdivide":      "linear",
    "checkerboard_remesh":  "linear",
    "ds_bc_new_subdivide":  "linear",
    "dome_subdivide":       "pending",
    "create_crust":         "crust",
}
for _op in OPS:
    _op.diff = _DIFF_BY_NAME[_op.name]


# ─────────────────────────────────────────────────────────────────────────────
# Markdown generation
# ─────────────────────────────────────────────────────────────────────────────

MD_HEADER = """# TopMod Operator Reference

> Generated by `scripts/generate_op_gallery.py` — manual edits will be
> overwritten. Edit the registry in that script and re-run it.

Complete reference for every mesh operator in GenesisTopmod: signature,
parameters, closed-form oracle (element-count effect), tokenizer opcode, and
a before/after visualization. Every operator preserves 2-manifoldness at
every step (the constructive DLFL guarantee of Akleman & Chen 2003).

- The **oracle** column gives the exact closed-form effect on (V, E, F); it
  is what `tests/test_semantic_oracle.py` asserts. All operators preserve χ
  and genus except `add_handle` and crust hole punching.
- The **token** column is the opcode in the `topmod/tokenizer.py` vocabulary;
  tokenized operators can be serialized to integer-ID sequences and replayed
  by `detokenize` (the basis for autoregressive mesh generation).
- All images are rendered by this script; both sides are annotated with
  V/E/F counts.
- The **Diff** column reports PyTorch differentiability of the operator's
  *position map* (output vertex coordinates as a function of input
  coordinates, with the operator sequence held fixed) via
  `topmod/diffgeo.py`. Topology itself is always discrete and carries no
  gradient. ✅ = supported today (18 traced/implemented ops + 3
  identity-geometry ops + free-parameter positions), ⏳ = planned phase 2
  (normal-based schemes and local single-element operators). See
  `docs/diffgeo.md` for the API.

## Quick Reference

| # | Operator | Token | Oracle (V', E', F') | Diff | Image |
|---|---|---|---|---|---|
"""

MD_USAGE_FOOTER = """
## Tokenizer Usage

```python
from topmod import tokenize, detokenize, build_vocabulary, encode_sequence
from topmod.tokenizer import TopModToken

# Every token sequence executes from the deterministic icosahedron primitive
tokens = [TopModToken(op='PENT'), TopModToken(op='DUAL'), TopModToken(op='EOS')]
mesh = detokenize(tokens)            # guaranteed: is_manifold(mesh) == True

vocab = build_vocabulary()           # token → integer ID (append-only)
ids = encode_sequence(tokens, vocab)
```

Holed shells (a genus construction that generative models can learn):

```python
# After CRUST the mirrored face pairs have deterministic ordinals:
# outer face i ↔ inner face F+i, so hole punching is expressible with the
# existing HDL(face1, face2) token.
tokens = [TopModToken(op='CRUST'),
          TopModToken(op='HDL', refs=(0, 20)),   # icosahedron: F=20
          TopModToken(op='EOS')]
```

## Differentiable Geometry Usage

Operators marked ✅ in the Diff column can be composed into an end-to-end
differentiable map (topology fixed, gradients w.r.t. base-primitive
positions and, for `create_crust`, its `thickness`):

```python
import torch
from topmod.diffgeo import DiffSequence

seq = DiffSequence("cube").append("DS").append("CC") \
                          .append("CRUST", thickness=0.1)
final_verts = seq.forward()      # differentiable w.r.t. seq.verts0
tris        = seq.triangles()    # for nvdiffrast / pipeline.geometry_optimizer
(final_verts ** 2).sum().backward()
```

Full API and correctness contract: `docs/diffgeo.md`.

## Testing

```bash
python3 -m pytest tests/ -q --ignore=tests/test_manifold_loss.py --ignore=tests/test_pipeline.py
```

Per-operator oracle tests live in `tests/test_semantic_oracle.py` (four
primitives × exact ΔV/ΔE/ΔF/χ/genus + face-degree census); token tests in
`tests/test_tokenizer.py`.

## References

- Akleman & Chen 2003 — the minimal complete DLFL operator set
- `docs/reference_semantics.md` — clean-room extraction (with χ
  verification) of the reference library's semantics (davyrisso/topmod3d, GPL)
- `docs/vocabulary_roadmap.md` — vocabulary evolution roadmap
"""


def _anchor(i: int, name: str) -> str:
    # GitHub slug for a heading like "### #3 insert_edge" is "3-insert_edge"
    return f"{i}-{name.lower()}"


def gen_markdown() -> None:
    lines = [MD_HEADER]
    for i, op in enumerate(OPS, start=1):
        img = (f"[img](assets/ops/{op.name}.png)"
               if not op.no_image else "—")
        diff_short = _DIFF[op.diff][0]
        lines.append(f"| #{i} | [`{op.name}`](#{_anchor(i, op.name)}) "
                     f"| {op.token} | {op.oracle} | {diff_short} | {img} |\n")

    cat = None
    for i, op in enumerate(OPS, start=1):
        if op.category != cat:
            cat = op.category
            lines.append(f"\n---\n\n## {cat}\n")
        lines.append(f"\n### #{i} {op.name}\n\n")
        lines.append(f"{op.desc}\n\n")
        lines.append(f"- **Signature**: `{op.signature}`\n")
        lines.append(f"- **Token**: `{op.token}`\n")
        lines.append(f"- **Oracle**: {op.oracle}\n")
        lines.append(f"- **Parameters**: {op.params}\n")
        diff_short, diff_long = _DIFF[op.diff]
        lines.append(f"- **Differentiable (PyTorch)**: {diff_short} — "
                     f"{diff_long}\n")
        lines.append(f"- **Example primitive**: {op.base_name}\n\n")
        lines.append("```python\n" + op.example + "\n```\n")
        if op.no_image:
            lines.append(f"\n*(no image: {op.no_image_reason})*\n")
        else:
            lines.append(f"\n![{op.name}](assets/ops/{op.name}.png)\n")

    lines.append(MD_USAGE_FOOTER)
    with open(MD_PATH, "w") as fh:
        fh.write("".join(lines))
    print(f"wrote {MD_PATH}")


def gen_images() -> None:
    os.makedirs(ASSET_DIR, exist_ok=True)
    for op in OPS:
        if op.no_image or op.apply is None:
            continue
        before = op.base()
        after = op.apply(op.base())
        out = render_pair(op.name, before, after, alpha_after=op.alpha_after)
        print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--md-only", action="store_true")
    args = ap.parse_args()
    if not args.md_only:
        gen_images()
    gen_markdown()
