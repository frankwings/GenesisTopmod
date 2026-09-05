"""External baseline: Palfinger 2022 'Continuous Remeshing for Inverse Rendering' (original code,
Resource/continuous-remeshing) run under OUR 64-view star setup and scored with OUR held-out exam.
Fairness: same star cameras (8 az x 8 el, distance 2 x radius, fov 2*atan(0.5)), same GT mesh, same
image size (RES, default 256). Its supervision is its own: normal images + alpha (it is what the
method is built on). Optimisation happens in its unit-sphere frame (its length parameters and lr are
in that frame), output is scaled back to our frame for the exam.
Run: MODE=64v SHAPE=fertility STEPS=1200 python3 despike/run_palfinger.py
"""
import sys, os, time
ROOT = "/home/kingy/Projects/Genesis/GenesisTopmod"
for p in (f"{ROOT}/experiments/opseq_v5", f"{ROOT}/experiments/opseq_v5/despike", ROOT, f"{ROOT}/Resource/continuous-remeshing"):
    sys.path.insert(0, p)
os.chdir(f"{ROOT}/experiments/opseq_v5"); os.environ.setdefault("MODE", "64v")
import shutil
_shim = f"{ROOT}/Resource/continuous-remeshing/torch_scatter.py"
if not os.path.exists(_shim): shutil.copy(f"{ROOT}/experiments/opseq_v5/despike/torch_scatter_shim.py", _shim)  # Resource/ is not versioned
import numpy as np, torch, nvdiffrast.torch as dr
import cow_v13
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight
from dlfl_untangle import si_faces, fold_frac
import run_64v
from core.opt import MeshOptimizer
from core.remesh import calc_vertex_normals
from util.func import make_star_cameras, make_sphere
from util.render import NormalsRenderer

SHAPE = os.environ.get("SHAPE", "fertility"); STEPS = int(os.environ.get("STEPS", "1200"))
RES = int(os.environ.get("RES", "256")); TAG = os.environ.get("TAG", f"{SHAPE}_palfinger{STEPS}")
LEVEL = int(os.environ.get("SPHERE_LEVEL", "2")); OUTD = "/tmp/liou_cow_viz"

# ---- our exam setup (identical to phase5_taubin) ----
ctx = dr.RasterizeCudaContext()
gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
gvn = normalize_to_range(gv); maxr = float(np.linalg.norm(gvn, axis=1).max())
mvps, views = run_64v.star_cameras(maxr)
gt, gtd, _, _ = run_64v.make_gt(ctx, mvps, views, SHAPE); cow_v13.N_VIEWS = 64
p1b._MVPS, p1b._GT = mvps, gt; p1b.SHAPE = SHAPE
px = 2 * maxr / 256.0

def report(tag, V, F):
    F = np.asarray(F, np.int64); ho = heldout_exam(ctx, V, F); wt, _ = check_watertight(F); s = si_faces(V, F)
    E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)
    genus = (2 - (len(V) - len(E) + len(F))) // 2; me = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()
    print(f"[{tag}] V={len(V)} F={len(F)} watertight={wt} genus={genus} | ho16={ho[0]:.4f} hair={ho[1]} maxblob={ho[2]} "
          f"| SI={100*s/len(F):.1f}% folds={100*fold_frac(V, F):.1f}% | mean edge {me/px:.1f} px", flush=True)

# ---- Palfinger in its unit-sphere frame ----
tv = torch.tensor(gvn / maxr, dtype=torch.float32, device="cuda"); tf = torch.tensor(gf, dtype=torch.long, device="cuda")
from util.func import _projection
mv, _ = make_star_cameras(8, 8, distance=2.0)                       # our star: distance 2R
# their default projection has near=1.0: at distance 2 an object of radius 1 touches the near plane
# (front clipped). Use near 0.1 and keep fov 2*atan(0.5): half-width at near = 0.5*near.
proj = _projection(0.05, "cuda", n=0.1, f=50.0)
renderer = NormalsRenderer(mv, proj, [RES, RES])
target = renderer.render(tv, calc_vertex_normals(tv, tf), tf)
verts, faces = make_sphere(level=LEVEL, radius=0.5)
opt = MeshOptimizer(verts, faces); verts = opt.vertices
t0 = time.time()
for i in range(STEPS):
    opt.zero_grad()
    normals = calc_vertex_normals(verts, faces)
    imgs = renderer.render(verts, normals, faces)
    loss = (imgs - target).abs().mean()
    loss.backward(); opt.step()
    verts, faces = opt.remesh()
    if (i + 1) % 200 == 0 or i + 1 == STEPS:
        V = verts.detach().cpu().numpy().astype(np.float64) * maxr; F = faces.cpu().numpy().astype(np.int64)
        print(f"[palfinger step {i+1}/{STEPS}] loss={loss.item():.4f} V={len(V)} F={len(F)} ({time.time()-t0:.0f}s)", flush=True)
        report(f"palfinger {i+1}", V, F)
V = verts.detach().cpu().numpy().astype(np.float64) * maxr; F = faces.cpu().numpy().astype(np.int64)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=F)
print(f"[palfinger] wall {time.time()-t0:.0f}s saved cow_{SHAPE}_{TAG}.npz", flush=True)
