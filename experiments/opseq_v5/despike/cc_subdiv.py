"""TopMod Catmull-Clark as the coarse-to-fine subdivision of Stage 1 (run_64v C2F_SUBDIV=cc).
Stage 1 keeps the quad mesh (cc2 -> cc3 = 1 quad -> 4 quads, rendered through a fixed fan triangulation);
once the mesh is triangles-only (after the DLFL clean loop / genus stage) CC is applied to the triangle
mesh: 1 tri -> 3 quads -> 6 tris. Unlike the numpy midpoint split, CC also MOVES every vertex toward the
smooth limit surface (approximating scheme), so each level starts from a fairer surface.
"""
import sys, numpy as np
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
from topmod.primitives import make_icosahedron, _build_mesh
from topmod.subdivision import catmull_clark
from topmod.diffgeo import mesh_to_arrays
from topmod.high_level_ops import triangulate_all
from topmod.io import to_triangle_arrays
from topmod.validate import check_all

def tris_of(v, polys):
    """Render triangulation of the polygon mesh via TopMod triangulate_all (DLFL insert_edge fan); vertex order kept."""
    m = _build_mesh([tuple(map(float, p)) for p in np.asarray(v)], [list(map(int, f)) for f in polys])
    triangulate_all(m)
    ok, errs = check_all(m); assert ok, errs
    _, tris = to_triangle_arrays(m)
    return np.asarray(tris, np.int32)

def _norm(v):
    mn, mx = float(v.min()), float(v.max())
    return (v - (mn + mx) / 2.0) * (2.0 / max(mx - mn, 1e-6))

def icosphere_cc2():
    m = make_icosahedron(); m = catmull_clark(m); m = catmull_clark(m)
    pos, polys = mesh_to_arrays(m)
    v = _norm(np.array(pos, np.float64)); return v, polys, tris_of(v, polys)

def cc_subdivide(v, polys):
    """One TopMod Catmull-Clark round on (positions, polygon faces). Returns (v', polys', tris')."""
    m = _build_mesh([tuple(map(float, p)) for p in np.asarray(v)], [list(map(int, f)) for f in polys])
    m2 = catmull_clark(m)
    ok, errs = check_all(m2); assert ok, errs
    pos, polys2 = mesh_to_arrays(m2)
    v2 = np.array(pos, np.float64); return v2, polys2, tris_of(v2, polys2)
