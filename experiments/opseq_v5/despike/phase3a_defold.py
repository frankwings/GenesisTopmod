"""Phase 3a: de-fold -- remove buried / self-intersecting material with DLFL
collapse_edge_tri, keeping the mesh a closed 2-manifold at every step.

Boss (2026-09-02) spotted that the watertight results look "too smooth":
far-view renders hide crumpled, folded faces pressed inside the surface
(collapsed webbing, sliver fans). Measured: 46-83% of faces are in
self-intersections, 8-17% of edges are folds (>120 deg). IoU never sees them.

Strategy (topology-preserving, TopMod-only):
  1. BURIED faces = never hit by the rasterizer from 64 star directions at
     1024px. They contribute nothing to any silhouette -> removable for free.
  2. Collapse their edges with DLFL collapse_edge_tri (link-condition guard,
     Euler preserved, V-1/E-3/F-2 per collapse). Folded material shrinks to
     points and disappears.
  3. Short DR settle to re-fit, repeat.
Report buried count, self-intersecting face %, fold-edge %, ho16 each round.

Run: MODE=6v TAG=p3a6 BASE_NPZ=... python3 despike/phase3a_defold.py
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
os.environ.setdefault("MODE", "6v")

import time, tempfile, collections
import numpy as np, torch
import open3d as o3d
import nvdiffrast.torch as dr

import cow_v13
from cow_v13 import DEVICE
from eval_local_refine import (setup_scene, render_views_n, compute_iou_n,
                               load_obj, normalize_to_range, BUNNY_PATH)
from pipeline.cameras import transform_to_clip
from escape_util import escape_mask
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight, settle, _set_faces
from topmod.io import from_obj, to_triangle_arrays
from topmod.high_level_ops import collapse_edge_tri

SHAPE = os.environ.get("SHAPE", "armadillo")
TAG = os.environ.get("TAG", "p3a6")
MODE = os.environ.get("MODE", "6v")
BASE_NPZ = os.environ.get("BASE_NPZ", "/tmp/liou_cow_viz/cow_armadillo_p2a6_it5_best.npz")
ROUNDS = int(os.environ.get("ROUNDS", "4"))
SETTLE_STEPS = int(os.environ.get("SETTLE_STEPS", "150"))
VIS_RES = int(os.environ.get("VIS_RES", "1024"))
MAX_COLLAPSE = int(os.environ.get("MAX_COLLAPSE", "1500"))   # per round
# Phase 3b: interior-only collapses. 3a collapsed the shortest edge of each
# buried face, which often joins a buried vertex to a VISIBLE one -> midpoint
# collapse dragged the outer surface (ho16 -1.3, hair x4). Cross-sections
# show the defect is a redundant inner/coincident skin (pocket folded inward).
# 3b: (A) collapse edges whose both endpoints are fully buried (pocket shrinks
# to a point, outer surface untouched); (B) pocket-mouth edges (buried +
# visible endpoint) collapse with the survivor SNAPPED to the visible position.
INTERIOR_ONLY = os.environ.get("INTERIOR_ONLY", "0") == "1"
OUTD = "/tmp/liou_cow_viz"

torch.manual_seed(0); np.random.seed(0)
if MODE == "64v":
    import run_64v
    ctx = dr.RasterizeCudaContext()
    gv, _gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
    max_r = float(np.linalg.norm(normalize_to_range(gv), axis=1).max())
    mvps, views = run_64v.star_cameras(max_r)
    gt, gtd, _gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
    run_64v._MVPS, run_64v._GT = mvps, gt
    cow_v13.N_VIEWS = 64
else:
    scene = setup_scene(SHAPE, DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]
p1b._MVPS, p1b._GT = mvps, gt
p1b.SHAPE = SHAPE
# visibility directions: always the 64 star rig (independent of supervision)
import run_64v as _r64
gv_, _ = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
vis_mvps, _ = _r64.star_cameras(float(np.linalg.norm(normalize_to_range(gv_), axis=1).max()))

z = np.load(BASE_NPZ)
V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)


# ---------------------------------------------------------------- metrics
def buried_faces(V, Fa):
    vt = torch.tensor(V, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(Fa.astype(np.int32), dtype=torch.int32, device=DEVICE)
    seen = torch.zeros(len(Fa), dtype=torch.bool, device=DEVICE)
    with torch.no_grad():
        for i in range(len(vis_mvps)):
            pos = transform_to_clip(vt, vis_mvps[i])
            rast, _ = dr.rasterize(ctx, pos, ft, resolution=[VIS_RES, VIS_RES])
            ids = rast[0, :, :, 3].long() - 1
            seen[ids[ids >= 0]] = True
    return (~seen).cpu().numpy()


def si_faces(V, Fa):
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, float)),
                                   o3d.utility.Vector3iVector(np.asarray(Fa, np.int32)))
    pairs = np.asarray(om.get_self_intersecting_triangles())
    return len(np.unique(pairs)) if len(pairs) else 0


def fold_frac(V, Fa):
    n = np.cross(V[Fa[:, 1]] - V[Fa[:, 0]], V[Fa[:, 2]] - V[Fa[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ef = collections.defaultdict(list)
    for i, (a, b, c) in enumerate(Fa):
        for e in ((a, b), (b, c), (c, a)):
            ef[(min(e), max(e))].append(i)
    cosang = np.array([n[f[0]] @ n[f[1]] for f in ef.values() if len(f) == 2])
    return float((cosang < -0.5).mean())          # > 120 deg


def report(tag, V, Fa):
    ho = heldout_exam(ctx, V, Fa)
    wt, nbad = check_watertight(Fa)
    bur = buried_faces(V, Fa).sum(); sif = si_faces(V, Fa)
    print(f"[{tag}] V={len(V)} F={len(Fa)} watertight={wt} | ho16={ho[0]:.4f} "
          f"hair={ho[1]} | buried={bur} ({100*bur/len(Fa):.1f}%) "
          f"SI faces={sif} ({100*sif/len(Fa):.1f}%) folds={100*fold_frac(V, Fa):.1f}%",
          flush=True)
    return ho[0]


# ---------------------------------------------------------------- DLFL collapse
def dlfl_collapse_faces(V, Fa, target_faces, max_collapse):
    """Collapse the shortest edge of each target face (smallest first).
    One DLFL session; face/edge objects checked live. Returns V2, F2, n."""
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    try:
        mesh = from_obj(path)
    finally:
        os.unlink(path)
    faces = list(mesh.iter_faces())
    assert len(faces) == len(Fa)
    tri = V[Fa]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    order = sorted(target_faces, key=lambda i: area[i])
    n = 0
    for fi in order:
        f = faces[fi]
        if f.id not in mesh.faces: continue          # already consumed
        hes = f.halfedges()
        if len(hes) != 3: continue
        def elen(h):
            a, b = h.origin, h.twin.origin if h.twin else h.origin
            return (a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2
        for h in sorted(hes, key=elen):
            if h.edge.id not in mesh.edges: continue
            if collapse_edge_tri(mesh, h.edge) is not None:
                n += 1; break
        if n >= max_collapse: break
    vv, ff = to_triangle_arrays(mesh)
    return np.asarray(vv, float), np.asarray(ff, np.int64), n


def dlfl_collapse_interior(V, Fa, buried, max_collapse):
    """3b: buried vertex = every incident face buried. Phase A collapses
    buried-buried edges (shortest first); Phase B collapses buried-visible
    edges snapping the survivor onto the visible endpoint."""
    nv = len(V)
    inc_all = np.ones(nv, bool); inc_any = np.zeros(nv, bool)
    for i, f in enumerate(Fa):
        for v in f:
            inc_all[v] &= buried[i]; inc_any[v] |= buried[i]
    bvert = inc_all                               # fully buried vertices
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    try:
        mesh = from_obj(path)
    finally:
        os.unlink(path)
    verts = list(mesh.vertices.values()); assert len(verts) == nv
    vid2idx = {v.id: i for i, v in enumerate(verts)}
    def endpoints(e):
        a = e.he0.origin; b = e.he1.origin
        return vid2idx.get(a.id), vid2idx.get(b.id), a, b
    nA = nB = 0
    edges = list(mesh.edges.values())
    def elen(e):
        _, _, a, b = endpoints(e)
        return (a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2
    # Phase A: buried-buried
    for e in sorted(edges, key=elen):
        if e.id not in mesh.edges: continue
        ia, ib, a, b = endpoints(e)
        if ia is None or ib is None or not (bvert[ia] and bvert[ib]): continue
        if collapse_edge_tri(mesh, e) is not None:
            nA += 1
            if nA >= max_collapse: break
    # Phase B: buried-visible, snap survivor to the visible endpoint
    for e in sorted(list(mesh.edges.values()), key=elen):
        if e.id not in mesh.edges: continue
        ia, ib, a, b = endpoints(e)
        if ia is None or ib is None: continue
        if bvert[ia] == bvert[ib]: continue
        vis = b if bvert[ia] else a
        vx, vy, vz = vis.x, vis.y, vis.z
        surv = collapse_edge_tri(mesh, e)
        if surv is not None:
            surv.x, surv.y, surv.z = vx, vy, vz
            nB += 1
            if nA + nB >= 2 * max_collapse: break
    vv, ff = to_triangle_arrays(mesh)
    return np.asarray(vv, float), np.asarray(ff, np.int64), nA, nB


def escape_fn(Vnp):
    vt = torch.tensor(np.asarray(Vnp), dtype=torch.float32, device=DEVICE)
    return escape_mask(vt, mvps, gt, dilate=2).cpu().numpy()


ho0 = report("base", V, Fa)
best = (ho0, V.copy(), Fa.copy(), -1)
t0 = time.time()
for rnd in range(ROUNDS):
    bmask = buried_faces(V, Fa)
    bur = np.where(bmask)[0]
    if len(bur) == 0:
        print(f"[r{rnd}] no buried faces left", flush=True); break
    if INTERIOR_ONLY:
        V, Fa, nA, nB = dlfl_collapse_interior(V, Fa, bmask, MAX_COLLAPSE)
        n = nA + nB
        print(f"[r{rnd}] DLFL interior collapses A(buried-buried)={nA} "
              f"B(mouth, snapped)={nB} of {len(bur)} buried faces -> "
              f"V={len(V)} F={len(Fa)} ({time.time()-t0:.0f}s)", flush=True)
        if n == 0:
            print(f"[r{rnd}] no collapsible interior edges", flush=True); break
    else:
        V, Fa, n = dlfl_collapse_faces(V, Fa, bur.tolist(), MAX_COLLAPSE)
    wt, nbad = check_watertight(Fa)
    print(f"[r{rnd}] DLFL collapsed {n} edges targeting {len(bur)} buried faces -> "
          f"V={len(V)} F={len(Fa)} watertight={wt} ({time.time()-t0:.0f}s)", flush=True)
    assert wt, f"non-manifold after collapse: {nbad}"
    if SETTLE_STEPS > 0:
        _set_faces(Fa)
        V, iou_s = settle(ctx, V, Fa, gt, gtd, mvps, SETTLE_STEPS, escape_fn)
        V = np.asarray(V, np.float64)
    ho = report(f"r{rnd}", V, Fa)
    if ho >= best[0] - 0.002:      # accept small IoU cost for cleaner geometry
        best = (ho, V.copy(), Fa.copy(), rnd)

print(f"\n[p3a] base ho16={ho0:.4f} -> final {report('final', V, Fa):.4f}", flush=True)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
with open(f"{OUTD}/cow_{SHAPE}_{TAG}.obj", "w") as fh:
    for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
    for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
print(f"[p3a] saved cow_{SHAPE}_{TAG}.npz/.obj", flush=True)
