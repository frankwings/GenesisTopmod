"""Nicolet, Jacobson, Jakob 2021 "Large Steps in Inverse Rendering of Geometry" (official largesteps
package: (I + lambda L) parameterization + AdamUniform + periodic Botsch-Kobbelt remeshing) run on OUR
setup: same 64 star views @256^2, same image supervision as our Stage-4 (silhouette L1 + masked depth
L1 x W_DEPTH + headlight-diffuse L1 x W_DIFF), same ho16 exam. Fixed topology: starts from an
icosphere (genus 0), Botsch-Kobbelt remeshing keeps it. Their scene renderer / SH lighting is replaced
by our render_sdd so the comparison isolates the optimizer + remeshing, not the lighting model.
Params from their figures/remeshing: alpha=0.95, AdamUniform lr 1e-2, L1 loss, remesh h = 0.5 x mean
edge, step size x0.8 after each remesh.
Env: SHAPE, STEPS (1200), REMESH_AT ("400,800"), LR (1e-2), ALPHA (0.95), SPHERE_LEVEL (3), TAG."""
import sys, os, time
ROOT = "/home/kingy/Projects/Genesis/GenesisTopmod"
for p in (f"{ROOT}/experiments/opseq_v5", f"{ROOT}/experiments/opseq_v5/despike", ROOT, f"{ROOT}/Resource/large-steps-pytorch"):
    sys.path.insert(0, p)
os.chdir(f"{ROOT}/experiments/opseq_v5"); os.environ.setdefault("MODE", "64v")
import numpy as np, torch, torch.nn.functional as F
import nvdiffrast.torch as dr
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
from eval_extrude_v3 import depth_loss_masked, W_DEPTH
import cow_v13
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight
from dlfl_untangle import si_faces, fold_frac
import run_64v
from run_64v import render_sdd
from largesteps.geometry import compute_matrix
from largesteps.optimize import AdamUniform
from largesteps.parameterize import to_differential, from_differential
from gpytoolbox import remesh_botsch, icosphere

SHAPE = os.environ.get("SHAPE", "kitten"); STEPS = int(os.environ.get("STEPS", "1200"))
TAG = os.environ.get("TAG", f"{SHAPE}_nicolet{STEPS}"); OUTD = "/tmp/liou_cow_viz"
REMESH_AT = [int(x) for x in os.environ.get("REMESH_AT", "400,800").split(",") if x]
LR = float(os.environ.get("LR", "1e-2")); ALPHA = float(os.environ.get("ALPHA", "0.95"))
LEVEL = int(os.environ.get("SPHERE_LEVEL", "3")); W_DIFF = float(os.environ.get("W_DIFF", "1.0"))
DEVICE = "cuda"; NV = 64
SOLVER = os.environ.get("SOLVER", "CG")   # cholespy 1.0.0 segfaults with torch 2.12+cu130; their CG solver is exact to 1e-6

ctx = dr.RasterizeCudaContext()
gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
gvn = normalize_to_range(gv); maxr = float(np.linalg.norm(gvn, axis=1).max())
mvps, views = run_64v.star_cameras(maxr)
gt, gtd, gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE); cow_v13.N_VIEWS = 64
p1b._MVPS, p1b._GT = mvps, gt; p1b.SHAPE = SHAPE
px = 2 * maxr / 256.0
targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
gtd_t = [torch.from_numpy(np.asarray(gtd[i], np.float32)).to(DEVICE) for i in range(NV)]
gtfg_t = [(targets[i, :, :, 0] > 0.5) for i in range(NV)]
gtdf_t = [torch.from_numpy(gtdiff[i]).float().to(DEVICE) for i in range(NV)]

def report(tag, V, Fa):
    V = np.ascontiguousarray(V, np.float64); Fa = np.ascontiguousarray(Fa, np.int64); ho = heldout_exam(ctx, V, Fa); wt, _ = check_watertight(Fa); s = si_faces(V, Fa)
    E = np.unique(np.sort(np.concatenate([Fa[:, [0, 1]], Fa[:, [1, 2]], Fa[:, [2, 0]]]), axis=1), axis=0)
    genus = (2 - (len(V) - len(E) + len(Fa))) // 2; me = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()
    print(f"[{tag}] V={len(V)} F={len(Fa)} watertight={wt} genus={genus} | ho16={ho[0]:.4f} hair={ho[1]} maxblob={ho[2]} "
          f"| SI={100*s/len(Fa):.1f}% folds={100*fold_frac(V, Fa):.1f}% | mean edge {me/px:.1f} px", flush=True)

def image_loss(vt, ft):
    sl = dl = fl = torch.tensor(0.0, device=DEVICE)
    for i in range(NV):
        sil, ndc_z, fg, diff = render_sdd(ctx, vt, ft, mvps[i], views[i])
        sl = sl + F.l1_loss(sil[0], targets[i]); dl = dl + depth_loss_masked(ndc_z, fg, gtd_t[i], gtfg_t[i])
        fl = fl + F.l1_loss(diff, gtdf_t[i])
    return (sl + W_DEPTH * dl + W_DIFF * fl) / NV, sl / NV

# init: icosphere of radius 0.5*maxr at the GT centroid (same frame as ours)
V0, F0 = icosphere(LEVEL); V0 = V0 * 0.5 * maxr + gvn.mean(0)
v = torch.tensor(V0, dtype=torch.float32, device=DEVICE); f = torch.tensor(F0.astype(np.int64), device=DEVICE)
def setup(v, f, lr):
    M = compute_matrix(v, f, lambda_=None, alpha=ALPHA)
    u = to_differential(M, v).detach().requires_grad_(True)
    return M, u, AdamUniform([u], lr=lr)
M, u, opt = setup(v, f, LR); lr = LR; t0 = time.time()
for it in range(STEPS):
    if it in REMESH_AT:
        with torch.no_grad():
            vc = from_differential(M, u, SOLVER).cpu().numpy().astype(np.double); fc = f.cpu().numpy().astype(np.int32)
            E = np.unique(np.sort(np.concatenate([fc[:, [0, 1]], fc[:, [1, 2]], fc[:, [2, 0]]]), axis=1), axis=0)
            h = float(np.linalg.norm(vc[E[:, 0]] - vc[E[:, 1]], axis=1).mean()) * 0.5
            vn, fn = remesh_botsch(vc, fc, 5, h, True)
        v = torch.tensor(vn, dtype=torch.float32, device=DEVICE); f = torch.tensor(fn.astype(np.int64), device=DEVICE)
        lr *= 0.8; M, u, opt = setup(v, f, lr)
        print(f"[nicolet remesh @{it}] h={h/px:.2f}px -> V={len(vn)} F={len(fn)} lr={lr:.4f}", flush=True)
    opt.zero_grad()
    v = from_differential(M, u, SOLVER)
    loss, sl = image_loss(v.contiguous(), f.to(torch.int32).contiguous())
    loss.backward(); opt.step()
    if (it + 1) % 200 == 0 or it + 1 == STEPS:
        V = np.ascontiguousarray(v.detach().cpu().numpy(), np.float64); Fa = np.ascontiguousarray(f.cpu().numpy(), np.int64)
        print(f"[nicolet step {it+1}/{STEPS}] loss={loss.item():.4f} sil={sl.item():.4f} V={len(V)} ({time.time()-t0:.0f}s)", flush=True)
        report(f"nicolet {it+1}", V, Fa)
V = np.ascontiguousarray(v.detach().cpu().numpy(), np.float64); Fa = np.ascontiguousarray(f.cpu().numpy(), np.int64)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
print(f"[nicolet] wall {time.time()-t0:.0f}s saved cow_{SHAPE}_{TAG}.npz", flush=True)
