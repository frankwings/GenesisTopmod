"""Phase 3d: untangle local flaps with DLFL edge flips (+ optional tangential
smoothing), no vertex-position optimization against images.

Diagnosis (2026-09-02): self-intersecting face pairs are LOCAL (90% within one
edge length, median 4-7 partners per face) -- folded-over flaps in the 1-ring,
not buried pockets (3a/3b) and not fixable by the adjacent-normal fold loss
alone (3c: x10 weight changed nothing). An edge flip is the classic local
untangling move and is pure DLFL: delete_edge (merge two triangles into a quad)
+ insert_edge between the opposite corners (split along the other diagonal).
Topology invariant: V, E, F unchanged, manifold preserved.

Criterion: flip a fold edge (adjacent normals dot < FOLD_COS) if the two
post-flip triangles are non-degenerate and less folded than before.

Run: MODE=6v TAG=p3d6 BASE_NPZ=... python3 despike/phase3d_flip.py
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
from eval_local_refine import setup_scene, load_obj, normalize_to_range, BUNNY_PATH
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight
from topmod.io import from_obj, to_triangle_arrays
from topmod.operators import insert_edge, delete_edge

SHAPE = os.environ.get("SHAPE", "armadillo")
TAG = os.environ.get("TAG", "p3d6")
MODE = os.environ.get("MODE", "6v")
BASE_NPZ = os.environ.get("BASE_NPZ", "/tmp/liou_cow_viz/cow_armadillo_p2a6_it5_best.npz")
PASSES = int(os.environ.get("PASSES", "6"))
FOLD_COS = float(os.environ.get("FOLD_COS", "0.0"))      # flip if n_a.n_b < this
SMOOTH_ITERS = int(os.environ.get("SMOOTH_ITERS", "0"))  # tangential smoothing after flips
SMOOTH_LAMBDA = float(os.environ.get("SMOOTH_LAMBDA", "0.3"))
OUTD = "/tmp/liou_cow_viz"

if MODE == "64v":
    import run_64v
    ctx = dr.RasterizeCudaContext()
    gv, _gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
    mvps, views = run_64v.star_cameras(float(np.linalg.norm(normalize_to_range(gv), axis=1).max()))
    gt, gtd, _d, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
    cow_v13.N_VIEWS = 64
else:
    scene = setup_scene(SHAPE, DEVICE)
    ctx, mvps, gt = scene["ctx"], scene["mvps"], scene["gt_uint8"]
p1b._MVPS, p1b._GT = mvps, gt
p1b.SHAPE = SHAPE


def si_faces(V, Fa):
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, float)),
                                   o3d.utility.Vector3iVector(np.asarray(Fa, np.int32)))
    pairs = np.asarray(om.get_self_intersecting_triangles())
    return len(np.unique(pairs)) if len(pairs) else 0


def fold_stats(V, Fa):
    n = np.cross(V[Fa[:, 1]] - V[Fa[:, 0]], V[Fa[:, 2]] - V[Fa[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ef = collections.defaultdict(list)
    for i, (a, b, c) in enumerate(Fa):
        for e in ((a, b), (b, c), (c, a)):
            ef[(min(e), max(e))].append(i)
    cs = np.array([n[f[0]] @ n[f[1]] for f in ef.values() if len(f) == 2])
    return float((cs < -0.5).mean()), float((cs < 0).mean())


def report(tag, V, Fa):
    ho = heldout_exam(ctx, V, Fa)
    wt, nbad = check_watertight(Fa)
    sif = si_faces(V, Fa); f120, f90 = fold_stats(V, Fa)
    print(f"[{tag}] V={len(V)} F={len(Fa)} watertight={wt} | ho16={ho[0]:.4f} hair={ho[1]} | "
          f"SI faces={sif} ({100*sif/len(Fa):.1f}%) folds>120={100*f120:.1f}% >90={100*f90:.1f}%",
          flush=True)
    return ho[0]


# ---------------------------------------------------------------- DLFL flip
def _tri_normal(a, b, c):
    n = np.cross([b.x-a.x, b.y-a.y, b.z-a.z], [c.x-a.x, c.y-a.y, c.z-a.z])
    l = np.linalg.norm(n)
    return n / l if l > 1e-14 else None, l


def try_flip(mesh, edge):
    he0, he1 = edge.he0, edge.he1
    fa, fb = he0.face, he1.face
    if fa is None or fb is None or fa.degree() != 3 or fb.degree() != 3:
        return False
    v0, v1 = he0.origin, he1.origin
    c = he0.prev.origin           # apex of fa (loop: v0->v1, v1->c, c->v0)
    d = he1.prev.origin           # apex of fb (loop: v1->v0, v0->d, d->v1)
    if c is d or c is v0 or c is v1 or d is v0 or d is v1:
        return False
    # current fold
    na, la = _tri_normal(v0, v1, c); nb, lb = _tri_normal(v1, v0, d)
    if na is None or nb is None: return False
    cur = float(na @ nb)
    if cur >= FOLD_COS: return False
    # c-d must not already be an edge (would create a duplicate edge)
    for h in c.outgoing_halfedges():
        if h.twin is not None and h.twin.origin is d: return False
    # post-flip triangles: (c, v0, d) and (d, v1, c)
    n1, l1 = _tri_normal(c, v0, d); n2, l2 = _tri_normal(d, v1, c)
    if n1 is None or n2 is None or l1 < 1e-3 * (la + lb) or l2 < 1e-3 * (la + lb):
        return False
    new = float(n1 @ n2)
    if new <= cur + 1e-6: return False
    merged = delete_edge(mesh, edge)
    hc = hd = None
    for h in merged.halfedges():
        if h.origin is c: hc = h
        if h.origin is d: hd = h
    if hc is None or hd is None:
        raise RuntimeError("flip: corners lost after delete_edge")
    insert_edge(mesh, hc, hd)
    return True


def dlfl_flip_pass(V, Fa, passes):
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    try:
        mesh = from_obj(path)
    finally:
        os.unlink(path)
    total = 0
    for p in range(passes):
        n = 0
        for e in list(mesh.edges.values()):
            if e.id not in mesh.edges: continue
            if try_flip(mesh, e): n += 1
        total += n
        print(f"    flip pass {p}: {n} flips", flush=True)
        if n == 0: break
    vv, ff = to_triangle_arrays(mesh)
    return np.asarray(vv, float), np.asarray(ff, np.int64), total


def tangential_smooth(V, Fa, iters, lam):
    V = V.copy(); nv = len(V)
    src = np.concatenate([Fa[:, 0], Fa[:, 1], Fa[:, 2], Fa[:, 1], Fa[:, 2], Fa[:, 0]])
    dst = np.concatenate([Fa[:, 1], Fa[:, 2], Fa[:, 0], Fa[:, 0], Fa[:, 1], Fa[:, 2]])
    deg = np.bincount(src, minlength=nv).astype(float)
    for _ in range(iters):
        cen = np.zeros_like(V); np.add.at(cen, src, V[dst]); cen /= np.maximum(deg, 1)[:, None]
        fn = np.cross(V[Fa[:, 1]] - V[Fa[:, 0]], V[Fa[:, 2]] - V[Fa[:, 0]])
        vn = np.zeros_like(V)
        for k in range(3): np.add.at(vn, Fa[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        d = cen - V
        d -= (d * vn).sum(1, keepdims=True) * vn        # tangential only
        V += lam * d
    return V


z = np.load(BASE_NPZ)
V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ho0 = report("base", V, Fa)
t0 = time.time()
V, Fa, nflip = dlfl_flip_pass(V, Fa, PASSES)
wt, nbad = check_watertight(Fa)
print(f"[flip] {nflip} DLFL edge flips total, watertight={wt} ({time.time()-t0:.0f}s)", flush=True)
assert wt
report("after-flip", V, Fa)
if SMOOTH_ITERS > 0:
    V = tangential_smooth(V, Fa, SMOOTH_ITERS, SMOOTH_LAMBDA)
    report(f"after-smooth{SMOOTH_ITERS}", V, Fa)
    V, Fa, nflip2 = dlfl_flip_pass(V, Fa, PASSES)
    wt, _ = check_watertight(Fa); assert wt
    report("after-smooth+flip", V, Fa)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
print(f"[p3d] saved cow_{SHAPE}_{TAG}.npz", flush=True)
