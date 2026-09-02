"""Phase 1i: core-fill -- morphological opening of missing-material evidence
as a localized 3D growth target.

Why 1h failed (measured): (a) visual hull is 3-6% fatter than GT at concavities
(physical floor, 20-config sweep) so "surface -> hull boundary" inflates the
whole body; (b) d_in (distance to nearest hull boundary) cannot see a finger
tube standing next to the fist: the fist skin is close to the tube's SIDE.

Boss's morphology idea applied to evidence: missing = hull & ~mesh_occ is a
fat skin (thin, everywhere) plus thick cores (fingers, unfilled body blocks).
ERODE by 3 voxels (opening) deletes the skin; surviving core is 96% true GT
material (top-5 blobs 100%). Core is recomputed each round as the mesh fills.

Loss on surface samples p within R voxels of the core (or inside it):
    C(p)            distance to nearest core voxel   -> pull INTO the core
  + relu(d_in - dz) distance to hull boundary inside -> push THROUGH the core
Outside-hull penalty retained (1f-c). DR losses keep authority.
Faces under active pull get DLFL subdivide_edge (real resolution). All topology
via TopMod DLFL, watertight asserted.

Run: TAG=p1i64 python3 despike/phase1i_corefill.py
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import time
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt, binary_erosion, label as cc_label
import open3d as o3d
import nvdiffrast.torch as dr

import run_64v
import cow_v13
from run_64v import star_cameras, make_gt, render_sdd, NV, _qual_loss
from cow_v13 import (build_adj, mean_edge_of, spike_pen, sliver_pen, DEVICE,
                     W_SPIKE, W_SLIVER)
from eval_local_refine import (load_obj, normalize_to_range, BUNNY_PATH,
                               depth_loss_masked, laplacian_loss,
                               edge_length_loss, render_views_n, compute_iou_n,
                               LR, LR_MIN, W_DEPTH, W_LAP, W_EDGE)
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight
from phase1c_pipeline import dlfl_subdivide_arrays
from hull_field import build_vote_hull

SHAPE = os.environ.get("SHAPE", "armadillo")
TAG = os.environ.get("TAG", "p1i64")
BASE_NPZ = os.environ.get("BASE_NPZ", "/tmp/liou_cow_viz/cow_armadillo_p1f64c.npz")
ROUNDS = int(os.environ.get("ROUNDS", "4"))
STEPS = int(os.environ.get("STEPS", "400"))
W_OUT = float(os.environ.get("W_OUT", "20.0"))
W_CORE = float(os.environ.get("W_CORE", "20.0"))
W_QUAL = float(os.environ.get("W_QUAL", "0.01"))
W_DIFF = float(os.environ.get("W_DIFF", "1.0"))
RAMP = int(os.environ.get("RAMP", "100"))
ANNEAL = int(os.environ.get("ANNEAL", "150"))
ERODE = int(os.environ.get("ERODE", "3"))          # opening radius (voxels)
R_GATE = float(os.environ.get("R_GATE", "4.0"))    # pull radius (voxels)
DEAD_IN_VOX = float(os.environ.get("DEAD_IN_VOX", "2.0"))
SUB_CAP = int(os.environ.get("SUB_CAP", "300"))
SUB_MIN_PEN = float(os.environ.get("SUB_MIN_PEN", "1.5"))  # voxels
OUTD = "/tmp/liou_cow_viz"

torch.manual_seed(0); np.random.seed(0)
ctx = dr.RasterizeCudaContext()
gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
gvn = normalize_to_range(gv)
mvps, views = star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
gt, gtd, gtdiff, _ = make_gt(ctx, mvps, views, SHAPE)
run_64v._MVPS, run_64v._GT = mvps, gt
cow_v13.N_VIEWS = 64
p1b._MVPS, p1b._GT = mvps, gt
p1b.SHAPE = SHAPE

z = np.load(BASE_NPZ)
V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ok, _ = check_watertight(Fa)
print(f"[p1i] base {BASE_NPZ} V={len(V)} F={len(Fa)} watertight={ok} "
      f"W_OUT={W_OUT} W_CORE={W_CORE} ERODE={ERODE} R_GATE={R_GATE} "
      f"rounds={ROUNDS}x{STEPS}", flush=True)

# tight hull: 1024px 2x-supersampled coverage>=0.25, vote 2 (hull/GT 1.059,
# GT erosion 0.00%)
HF = build_vote_hull(ctx, mvps, gvn, gf_gt, V, DEVICE,
                     nres=256, hires=1024, vote=2, ss_thr=0.25)
N = HF.nres; lo, hi, sp = HF.lo, HF.hi, HF.sp
DEAD_OUT = 1.0 * HF.pitch
DEAD_IN = DEAD_IN_VOX * HF.pitch
R_W = R_GATE * HF.pitch
axes = [np.linspace(lo[a], hi[a], N) for a in range(3)]
GRID = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)
print(f"[p1i] hull vox={HF.hull.sum()} pitch={HF.pitch:.4f}", flush=True)


def mesh_occupancy(Vv, Ff):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(np.asarray(Vv, np.float32)),
        o3d.core.Tensor(np.asarray(Ff, np.uint32))))
    out = np.zeros(len(GRID), bool)
    CH = 2_000_000
    for s in range(0, len(GRID), CH):
        out[s:s+CH] = sc.compute_occupancy(
            o3d.core.Tensor(GRID[s:s+CH])).numpy() > 0.5
    return out.reshape(N, N, N)


def build_core_field(Vv, Ff):
    """core = erode(hull & ~occ); returns C-field volume (dist to core) and
    core voxel count."""
    occ = mesh_occupancy(Vv, Ff)
    missing = HF.hull & ~occ
    core = binary_erosion(missing, iterations=ERODE)
    lab, n = cc_label(core, structure=np.ones((3, 3, 3), int))
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([])
    C = distance_transform_edt(~core, sampling=tuple(sp)).astype(np.float32)
    vol = torch.from_numpy(C).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return vol, int(core.sum()), n, (sorted(sizes.tolist(), reverse=True)[:5])


_BARY = torch.tensor([
    [1/3, 1/3, 1/3],
    [1/2, 1/2, 0.0], [0.0, 1/2, 1/2], [1/2, 0.0, 1/2],
    [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3],
], dtype=torch.float32, device=DEVICE)


def field_loss(verts_t, faces_l, Cvol):
    tri = verts_t[faces_l]
    pts = torch.einsum("sk,fkc->fsc", _BARY, tri).reshape(-1, 3)
    S = _BARY.shape[0]
    do = HF.dist(pts).view(-1, S)
    di = HF.dist_in(pts).view(-1, S)
    C = HF._sample(Cvol, pts).view(-1, S)
    gate = (C.detach() < R_W).float()
    core_pen = gate * (C + F.relu(di - DEAD_IN))
    pen = F.relu(do - DEAD_OUT) + (W_CORE / W_OUT) * core_pen
    area = 0.5 * torch.cross(tri[:, 1] - tri[:, 0],
                             tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    w = area.detach() / (area.detach().sum() + 1e-12)
    return (pen.mean(1) * w).sum(), core_pen.detach().mean(1), gate.mean(1)


def iou_fn(vv, ff):
    vt = torch.tensor(np.asarray(vv), dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(ff, np.int32), dtype=torch.int32, device=DEVICE)
    return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)


targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
gtd_t = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(NV)]
gtfg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(NV)]
gtdf_t = [torch.from_numpy(gtdiff[i]).float().to(DEVICE) for i in range(NV)]

ho0 = heldout_exam(ctx, V, Fa)
print(f"[p1i] BASE train={iou_fn(V, Fa):.4f} ho16={ho0[0]:.4f} hair={ho0[1]}",
      flush=True)
best = (ho0[0], V.copy(), Fa.copy(), -1)
t0 = time.time()

for rnd in range(ROUNDS):
    last = rnd == ROUNDS - 1
    Cvol, ncore, ncomp, top = build_core_field(V, Fa)
    print(f"[p1i r{rnd}] core={ncore} vox comps={ncomp} top5={top} "
          f"({time.time()-t0:.0f}s)", flush=True)
    verts_t = torch.tensor(V, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(Fa.astype(np.int32), dtype=torch.int32, device=DEVICE)
    faces_l = faces_t.long()
    src, dst, deg, excl = build_adj(Fa.astype(np.int32), len(V))
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR_MIN)
    for step in range(STEPS):
        opt.zero_grad()
        me = mean_edge_of(verts_t.detach(), src, dst)
        sl = dl = fl = torch.tensor(0.0, device=DEVICE)
        for i in range(NV):
            sil, ndc_z, fg, diff = render_sdd(ctx, verts_t, faces_t, mvps[i], views[i])
            sl = sl + F.l1_loss(sil[0], targets[i])
            dl = dl + depth_loss_masked(ndc_z, fg, gtd_t[i], gtfg_t[i])
            fl = fl + F.l1_loss(diff, gtdf_t[i])
        sl, dl, fl = sl / NV, dl / NV, fl / NV
        wh = W_OUT * min(1.0, (step + 1) / RAMP)
        if last and ANNEAL > 0 and step >= STEPS - ANNEAL:
            wh = wh * max(0.1, (STEPS - step) / ANNEAL)
        hl, fpen, fgate = field_loss(verts_t, faces_l, Cvol)
        loss = (sl + W_DEPTH * dl + W_DIFF * fl
                + W_LAP * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t)
                + W_QUAL * _qual_loss(verts_t, faces_t)
                + W_SPIKE * spike_pen(verts_t, src, dst, deg, me)
                + W_SLIVER * sliver_pen(verts_t, faces_l, me)
                + wh * hl)
        loss.backward()
        opt.step(); sched.step()
        if (step + 1) % 100 == 0:
            print(f"[p1i r{rnd}] step {step+1}/{STEPS} sil={sl.item():.4f} "
                  f"field={hl.item():.5f} gated_faces={(fgate > 0).sum().item()} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    V = verts_t.detach().cpu().numpy().astype(np.float64)
    ho_r = heldout_exam(ctx, V, Fa)
    print(f"[p1i r{rnd}] ho16={ho_r[0]:.4f} hair={ho_r[1]} V={len(V)}", flush=True)
    if ho_r[0] > best[0]:
        best = (ho_r[0], V.copy(), Fa.copy(), rnd)
    if last: break
    with torch.no_grad():
        _, fpen, fgate = field_loss(verts_t, faces_l, Cvol)
    cand = torch.where((fgate > 0) & (fpen > SUB_MIN_PEN * HF.pitch))[0].cpu().numpy()
    if len(cand) == 0:
        print(f"[p1i r{rnd}] no faces under pull, skip subdiv", flush=True)
        continue
    if len(cand) > SUB_CAP:
        cand = cand[np.argsort(fpen[cand].cpu().numpy())[::-1][:SUB_CAP]]
    V, Fa, ne = dlfl_subdivide_arrays(V, Fa, cand.tolist())
    ok, _ = check_watertight(Fa)
    print(f"[p1i r{rnd}] subdivided {ne} edges at {len(cand)} pulled faces "
          f"-> V={len(V)} F={len(Fa)} watertight={ok}", flush=True)
    assert ok

hof = heldout_exam(ctx, V, Fa)
print(f"\n[p1i] FINAL train={iou_fn(V, Fa):.4f} ho16={hof[0]:.4f} "
      f"hair={hof[1]} maxblob={hof[2]} V={len(V)} F={len(Fa)}")
print(f"[p1i] delta ho16: {ho0[0]:.4f} -> {hof[0]:.4f} ({(hof[0]-ho0[0])*100:+.2f}pts)")
print(f"[p1i] best round={best[3]} ho16={best[0]:.4f}", flush=True)
np.savez_compressed(os.path.join(OUTD, f"cow_{SHAPE}_{TAG}.npz"), verts=V, tris=Fa)
np.savez_compressed(os.path.join(OUTD, f"cow_{SHAPE}_{TAG}_best.npz"),
                    verts=best[1], tris=best[2])
print(f"[p1i] saved cow_{SHAPE}_{TAG}.npz (+_best)", flush=True)
