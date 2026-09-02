"""Phase 2a: reverse hybrid -- manifold-ize DMesh's our-init 6v result.

E1 showed DMesh@6v initialized from our manifold mesh reaches ho16 0.9705
(vs 0.5297 random init, vs ours 0.9160) but outputs a non-manifold soup
(631 bad edges). Here the soup becomes EVIDENCE: an unsigned distance field
to its surface (well-defined for any triangle soup) pulls our watertight mesh
toward it, with the 6-view DR losses (sil+depth) keeping authority and our
regularizers (lap/edge/qual/spike/sliver/fold/tube) keeping the mesh sane.
Result: DMesh-level geometry inside a guaranteed manifold, at 6 views.

Optimizer = phase1b.settle (cow_v13.optimize_phase, settle schedule) with the
target term injected through the same edge_length_loss hook settle uses for
L_qual. Faces still far from the target after a round get DLFL subdivide_edge.

Run: TAG=p2a6 python3 despike/phase2a_reverse.py
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
os.environ.setdefault("MODE", "6v")

import time
import numpy as np, torch
import torch.nn.functional as F
import open3d as o3d

import cow_v13
from cow_v13 import DEVICE
from eval_local_refine import (setup_scene, render_views_n, compute_iou_n,
                               W_EDGE)
from escape_util import escape_mask
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight, settle, _set_faces
from phase1c_pipeline import dlfl_subdivide_arrays
from eval_dmesh import load_any_mesh

SHAPE = os.environ.get("SHAPE", "armadillo")
TAG = os.environ.get("TAG", "p2a6")
BASE_NPZ = os.environ.get("BASE_NPZ", "/tmp/liou_cow_viz/cow_armadillo_p0_q0.01.npz")
TARGET_OBJ = os.environ.get("TARGET_OBJ",
    "/tmp/liou_cow_viz/dmesh6_ourinit_epoch_3_phase_1_last_aligned.obj")
ROUNDS = int(os.environ.get("ROUNDS", "3"))
STEPS = int(os.environ.get("STEPS", "400"))
W_T = float(os.environ.get("W_T", "20.0"))
NRES = int(os.environ.get("NRES", "256"))
DEAD_VOX = float(os.environ.get("DEAD_VOX", "0.5"))
SUB_THR_VOX = float(os.environ.get("SUB_THR_VOX", "2.0"))
SUB_CAP = int(os.environ.get("SUB_CAP", "200"))
DILATE = 2
OUTD = "/tmp/liou_cow_viz"

torch.manual_seed(0); np.random.seed(0)
scene = setup_scene(SHAPE, DEVICE)
ctx, mvps = scene["ctx"], scene["mvps"]
gt, gtd = scene["gt_uint8"], scene["gt_depths"]
p1b._MVPS, p1b._GT = mvps, gt
p1b.SHAPE = SHAPE

z = np.load(BASE_NPZ)
V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ok, _ = check_watertight(Fa)
tv, tf = load_any_mesh(TARGET_OBJ)
tv = np.asarray(tv, np.float32); tf = np.asarray(tf, np.uint32)
print(f"[p2a] base {BASE_NPZ} V={len(V)} watertight={ok} | target soup "
      f"V={len(tv)} F={len(tf)} | W_T={W_T} rounds={ROUNDS}x{STEPS}", flush=True)

# ---------------------------------------------------------------- target field
lo = np.minimum(tv.min(0), V.min(0)) - 0.03
hi = np.maximum(tv.max(0), V.max(0)) + 0.03
sp = (hi - lo) / (NRES - 1); PITCH = float(sp.max())
axes = [np.linspace(lo[a], hi[a], NRES) for a in range(3)]
G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
sc = o3d.t.geometry.RaycastingScene()
sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(tv), o3d.core.Tensor(tf)))
D = np.zeros(len(G), np.float32)
for s in range(0, len(G), 2_000_000):
    D[s:s+2_000_000] = sc.compute_distance(o3d.core.Tensor(G[s:s+2_000_000])).numpy()
vol = torch.from_numpy(D.reshape(NRES, NRES, NRES)).unsqueeze(0).unsqueeze(0).to(DEVICE)
lo_t = torch.tensor(lo, dtype=torch.float32, device=DEVICE)
hi_t = torch.tensor(hi, dtype=torch.float32, device=DEVICE)
DEAD = DEAD_VOX * PITCH
print(f"[p2a] target field NRES={NRES} pitch={PITCH:.4f} dead={DEAD:.4f}", flush=True)

def tdist(pts):
    g = 2.0 * (pts - lo_t) / (hi_t - lo_t) - 1.0
    grid = g[:, [2, 1, 0]].view(1, 1, 1, -1, 3)
    return F.grid_sample(vol, grid, mode="bilinear", padding_mode="border",
                         align_corners=True).view(-1)

_BARY = torch.tensor([
    [1/3, 1/3, 1/3], [1/2, 1/2, 0.0], [0.0, 1/2, 1/2], [1/2, 0.0, 1/2],
    [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3],
], dtype=torch.float32, device=DEVICE)

def target_loss(verts_t, faces_t):
    faces_l = faces_t.long()
    tri = verts_t[faces_l]
    pts = torch.einsum("sk,fkc->fsc", _BARY, tri).reshape(-1, 3)
    d = tdist(pts).view(-1, _BARY.shape[0])
    dv = tdist(verts_t)
    pen = F.relu(d - DEAD).mean(1)
    area = 0.5 * torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0],
                             dim=-1).norm(dim=-1)
    w = area.detach() / (area.detach().sum() + 1e-12)
    return (pen * w).sum() + F.relu(dv - DEAD).mean()

# inject through the same hook settle() uses (edge_length_loss * W_EDGE)
_orig_edge = cow_v13.edge_length_loss
def _edge_t(v, f):
    return _orig_edge(v, f) + (W_T / max(W_EDGE, 1e-9)) * target_loss(v, f)
cow_v13.edge_length_loss = _edge_t

def iou_fn(vv, ff):
    vt = torch.tensor(np.asarray(vv), dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(ff, np.int32), dtype=torch.int32, device=DEVICE)
    return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

def escape_fn(Vnp):
    vt = torch.tensor(np.asarray(Vnp), dtype=torch.float32, device=DEVICE)
    return escape_mask(vt, mvps, gt, dilate=DILATE).cpu().numpy()

def face_target_dist(Vv, Ff):
    with torch.no_grad():
        vt = torch.tensor(Vv, dtype=torch.float32, device=DEVICE)
        tri = vt[torch.tensor(Ff, device=DEVICE).long()]
        pts = torch.einsum("sk,fkc->fsc", _BARY, tri).reshape(-1, 3)
        return tdist(pts).view(-1, _BARY.shape[0]).mean(1).cpu().numpy()

ho0 = heldout_exam(ctx, V, Fa)
fd0 = face_target_dist(V, Fa)
print(f"[p2a] BASE train6={iou_fn(V, Fa):.4f} ho16={ho0[0]:.4f} hair={ho0[1]} "
      f"| face->target dist mean={fd0.mean():.4f} max={fd0.max():.4f} "
      f"far(>2vox)={(fd0 > 2*PITCH).sum()}", flush=True)
best = (ho0[0], V.copy(), Fa.copy(), -1)
t0 = time.time()
for rnd in range(ROUNDS):
    _set_faces(Fa)
    V, iou_s = settle(ctx, V, Fa, gt, gtd, mvps, STEPS, escape_fn)
    V = np.asarray(V, np.float64)
    ho_r = heldout_exam(ctx, V, Fa)
    fd = face_target_dist(V, Fa)
    print(f"[p2a r{rnd}] train6={iou_s:.4f} ho16={ho_r[0]:.4f} hair={ho_r[1]} "
          f"V={len(V)} | dist mean={fd.mean():.4f} far={(fd > 2*PITCH).sum()} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if ho_r[0] > best[0]:
        best = (ho_r[0], V.copy(), Fa.copy(), rnd)
    if rnd == ROUNDS - 1: break
    far = np.where(fd > SUB_THR_VOX * PITCH)[0]
    if len(far) == 0:
        print(f"[p2a r{rnd}] nothing far from target, skip subdiv", flush=True)
        continue
    if len(far) > SUB_CAP:
        far = far[np.argsort(fd[far])[::-1][:SUB_CAP]]
    V, Fa, ne = dlfl_subdivide_arrays(V, Fa, far.tolist())
    ok, _ = check_watertight(Fa)
    print(f"[p2a r{rnd}] DLFL subdivided {ne} edges at {len(far)} far faces -> "
          f"V={len(V)} F={len(Fa)} watertight={ok}", flush=True)
    assert ok

cow_v13.edge_length_loss = _orig_edge
hof = heldout_exam(ctx, V, Fa)
okf, _ = check_watertight(Fa)
print(f"\n[p2a] FINAL train6={iou_fn(V, Fa):.4f} ho16={hof[0]:.4f} hair={hof[1]} "
      f"maxblob={hof[2]} V={len(V)} F={len(Fa)} watertight={okf}")
print(f"[p2a] delta ho16: {ho0[0]:.4f} -> {hof[0]:.4f} ({(hof[0]-ho0[0])*100:+.2f}pts)"
      f" | best round={best[3]} ho16={best[0]:.4f}", flush=True)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}_best.npz", verts=best[1], tris=best[2])
with open(f"{OUTD}/cow_{SHAPE}_{TAG}.obj", "w") as fh:
    for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
    for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
print(f"[p2a] saved cow_{SHAPE}_{TAG}.npz/.obj (+_best)", flush=True)
