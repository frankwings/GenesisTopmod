"""Phase 5b: Taubin fairing as the final polish (positions only; connectivity
and therefore manifoldness untouched).

The 36k-face Phase 5 mesh scored 0.9914 but its back had a median dihedral
of 30.5 deg vs 10.1 deg for GT decimated to the same face count: pure
high-frequency jitter left by the optimizer (Laplacian x10 inside the loop
barely helped: 27.9). Taubin lambda|mu smoothing removes it without shrinkage:
  x5 : ho16 0.9957, hair 8,  back dihedral 7.9 deg, SI 1.4%
  x10: ho16 0.9944, hair 6,  back dihedral 6.3 deg
IoU goes UP because the jitter was silhouette noise too.

Run: MODE=64v ITERS=5 BASE_NPZ=... TAG=... python3 despike/phase5_taubin.py
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
os.environ.setdefault("MODE", "64v")
import numpy as np, collections, open3d as o3d, torch
import nvdiffrast.torch as dr
import cow_v13
from cow_v13 import DEVICE
from eval_local_refine import setup_scene, load_obj, normalize_to_range, BUNNY_PATH
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight
from dlfl_untangle import si_faces, fold_frac

SHAPE = os.environ.get("SHAPE", "armadillo")
MODE = os.environ.get("MODE", "64v")
ITERS = int(os.environ.get("ITERS", "5"))
LAM, MU = float(os.environ.get("LAM", "0.5")), float(os.environ.get("MU", "-0.53"))
BASE_NPZ = os.environ["BASE_NPZ"]
TAG = os.environ.get("TAG", f"taubin{ITERS}")
AUTO = int(os.environ.get("AUTO", "0"))
AUTO_MIN = int(os.environ.get("AUTO_MIN", "2"))  # floor: training IoU under-smooths (it rewards jitter that fits the training views)   # pick the iteration count that maximizes TRAINING-view IoU (no exam leakage)
ADAPTIVE = int(os.environ.get("ADAPTIVE", "0"))   # per-vertex strength scaled by local thickness (thin limbs smoothed less)
T0_EDGES = float(os.environ.get("T0_EDGES", "4.0"))  # thickness (in mean-edge units) at which full strength is reached
OUTD = "/tmp/liou_cow_viz"

if MODE == "64v":
    import run_64v
    ctx = dr.RasterizeCudaContext()
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
    gvn = normalize_to_range(gv)
    mvps, views = run_64v.star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
    gt, gtd, _, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
    cow_v13.N_VIEWS = 64
else:
    scene = setup_scene(SHAPE, DEVICE)
    ctx, mvps, gt = scene["ctx"], scene["mvps"], scene["gt_uint8"]
p1b._MVPS, p1b._GT = mvps, gt
p1b.SHAPE = SHAPE


def back_dihedral(V, F, ctr=(0.0, 0.2, 0.4), r0=0.6):
    V = np.asarray(V, float); F = np.asarray(F, np.int64); ctr = np.asarray(ctr)
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ef = collections.defaultdict(list)
    for i, (a, b, c) in enumerate(F):
        for e in ((a, b), (b, c), (c, a)):
            ef[(min(e), max(e))].append(i)
    a = [np.degrees(np.arccos(np.clip(n[fs[0]] @ n[fs[1]], -1, 1)))
         for (x, y), fs in ef.items() if len(fs) == 2 and np.linalg.norm((V[x] + V[y]) / 2 - ctr) < r0]
    return float(np.median(a)) if a else float("nan")


def report(tag, V, F):
    ho = heldout_exam(ctx, V, F); wt, _ = check_watertight(F); s = si_faces(V, F)
    print(f"[{tag}] V={len(V)} F={len(F)} watertight={wt} | ho16={ho[0]:.4f} hair={ho[1]} "
          f"maxblob={ho[2]} | SI={100*s/len(F):.1f}% folds={100*fold_frac(V, F):.1f}% "
          f"back_dihedral_med={back_dihedral(V, F):.1f}deg", flush=True)


d = np.load(BASE_NPZ)
V, F = d["verts"].astype(float), d["tris"].astype(np.int64)
report("base", V, F)
def local_thickness(V, F):
    """Distance from each vertex along -normal to the opposite side of the surface (ray cast on the
    mesh itself); the local feature size that a smoother must not erase."""
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    m.compute_vertex_normals(); n = np.asarray(m.vertex_normals)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.int32))))
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); me = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()
    org = (V - 1e-3 * me * n).astype(np.float32)          # start just inside
    rays = o3d.core.Tensor(np.concatenate([org, (-n).astype(np.float32)], 1))
    t = sc.cast_rays(rays)["t_hit"].numpy(); t[~np.isfinite(t)] = 1e9
    return t, me

def taubin_adaptive(V, F, iters, lam, mu, w):
    """Taubin lambda|mu with per-vertex strength w in [0,1] (uniform umbrella operator)."""
    V = V.copy(); nv = len(V)
    src = np.concatenate([F[:, 0], F[:, 1], F[:, 2], F[:, 1], F[:, 2], F[:, 0]])
    dst = np.concatenate([F[:, 1], F[:, 2], F[:, 0], F[:, 0], F[:, 1], F[:, 2]])
    deg = np.bincount(src, minlength=nv).astype(float)[:, None]
    for _ in range(iters):
        for k in (lam, mu):
            cen = np.zeros_like(V); np.add.at(cen, src, V[dst]); cen /= np.maximum(deg, 1)
            V += (k * w)[:, None] * (cen - V)
    return V

def train_iou(V, F):
    from cow_v13 import render_views_n, compute_iou_n
    vt = torch.tensor(np.asarray(V), dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(F, np.int32), dtype=torch.int32, device=DEVICE)
    return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

if AUTO:
    # data-driven strength: the smoothing that best matches the 64 TRAINING views. A coarse mesh
    # (fertility: mean edge 0.049) over-smooths at 5 iterations; a fine one (armadillo 0.032) wants 5.
    best = None
    for it in (0, 1, 2, 3, 5, 8):
        if it == 0: Vt = V.copy()
        else:
            m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
            Vt = np.asarray(m.filter_smooth_taubin(number_of_iterations=it, lambda_filter=LAM, mu=MU).vertices)
        tiou = train_iou(Vt, F)
        print(f"[auto] taubin x{it}: train IoU {tiou:.4f}", flush=True)
        if best is None or tiou > best[0] + 1e-5: best = (tiou, it, Vt)
    if best[1] < AUTO_MIN:
        m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
        best = (best[0], AUTO_MIN, np.asarray(m.filter_smooth_taubin(number_of_iterations=AUTO_MIN, lambda_filter=LAM, mu=MU).vertices))
    ITERS = best[1]; V2 = best[2]
    print(f"[auto] chosen ITERS={ITERS} (train IoU {best[0]:.4f})", flush=True)
elif ADAPTIVE:
    thick, me = local_thickness(V, F)
    w = np.clip(thick / (T0_EDGES * me), 0.0, 1.0)
    print(f"[adaptive] thickness median {np.median(thick[thick < 1e8]):.3f} (mean edge {me:.3f}); "
          f"verts damped (<{T0_EDGES} edges thick): {100 * (w < 1).mean():.1f}%, fully off (<1 edge): {100 * (thick < me).mean():.1f}%", flush=True)
    V2 = taubin_adaptive(V, F, ITERS, LAM, MU, w)
else:
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V), o3d.utility.Vector3iVector(F.astype(np.int32)))
    m = m.filter_smooth_taubin(number_of_iterations=ITERS, lambda_filter=LAM, mu=MU)
    V2 = np.asarray(m.vertices)
report(f"taubin x{ITERS}", V2, F)
if os.environ.get("SNAPSHOT_DIR"):
    import viz_snap
    viz_snap.snap(ctx, mvps, V, F, f"{os.environ.get('SNAPSHOT_TITLE', 'Taubin')} before", hold=30)
    viz_snap.snap(ctx, mvps, V2, F, f"{os.environ.get('SNAPSHOT_TITLE', 'Taubin')} x{ITERS} -> FINAL", hold=90)
out = f"{OUTD}/cow_{SHAPE}_{TAG}.npz"
np.savez_compressed(out, verts=V2, tris=F)
with open(out.replace(".npz", ".obj"), "w") as fh:
    for x, y, z in V2: fh.write(f"v {x} {y} {z}\n")
    for a, b, c in F: fh.write(f"f {a+1} {b+1} {c+1}\n")
print(f"saved {out} (+.obj)", flush=True)
