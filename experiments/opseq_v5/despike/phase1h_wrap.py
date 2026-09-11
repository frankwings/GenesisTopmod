"""Phase 1h: two-sided hull shrink-wrap + adaptive DLFL subdivision.

At 64 views the voting hull's boundary ~= GT surface (volume ratio 0.996).
Phase 1f used only the OUTSIDE half of that evidence (pull excess back in).
1h adds the INSIDE half: surface samples deep inside the hull mean missing
material above them (finger tubes standing empty) -> push the surface out
toward the hull boundary. DR losses (sil+depth+diffuse) keep full authority:
W_IN is small so the ~0.4% hull over-estimate at concavities is overridden
by image evidence.

Growth needs resolution: where surface is deep inside (d_in large), the mesh
is refined via DLFL subdivide_edge (phase1c.dlfl_subdivide_arrays) between
optimization rounds. All topology ops are TopMod DLFL; watertight asserted.

Run: TAG=p1h64 python3 despike/phase1h_wrap.py
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import time
import numpy as np, torch
import torch.nn.functional as F
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
TAG = os.environ.get("TAG", "p1h64")
BASE_NPZ = os.environ.get("BASE_NPZ", "/tmp/liou_cow_viz/cow_armadillo_p1f64c.npz")
WRAP_ROUNDS = int(os.environ.get("WRAP_ROUNDS", "3"))
STEPS = int(os.environ.get("STEPS", "500"))
W_OUT = float(os.environ.get("W_OUT", "20.0"))
W_IN = float(os.environ.get("W_IN", "5.0"))
W_QUAL = float(os.environ.get("W_QUAL", "0.01"))
W_DIFF = float(os.environ.get("W_DIFF", "1.0"))
RAMP = int(os.environ.get("RAMP", "100"))
ANNEAL = int(os.environ.get("ANNEAL", "200"))     # last round only
DEAD_OUT_VOX = 1.0
DEAD_IN_VOX = float(os.environ.get("DEAD_IN_VOX", "2.0"))
SUB_THR_VOX = float(os.environ.get("SUB_THR_VOX", "3.0"))  # d_in > thr -> subdiv
SUB_CAP = int(os.environ.get("SUB_CAP", "200"))   # max seed faces per round
OUTD = "/tmp/liou_cow_viz"

torch.manual_seed(0); np.random.seed(0)
ctx = dr.RasterizeCudaContext()
gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
gvn = normalize_to_range(gv)
max_r = float(np.linalg.norm(gvn, axis=1).max())
mvps, views = star_cameras(max_r)
gt, gtd, gtdiff, _ = make_gt(ctx, mvps, views, SHAPE)
run_64v._MVPS, run_64v._GT = mvps, gt
cow_v13.N_VIEWS = 64
p1b._MVPS, p1b._GT = mvps, gt
p1b.SHAPE = SHAPE

z = np.load(BASE_NPZ)
V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ok, _ = check_watertight(Fa)
print(f"[p1h] base {BASE_NPZ} V={len(V)} F={len(Fa)} watertight={ok} "
      f"W_OUT={W_OUT} W_IN={W_IN} rounds={WRAP_ROUNDS}x{STEPS}", flush=True)

HF = build_vote_hull(ctx, mvps, gvn, gf_gt, V, DEVICE)
DEAD_OUT = DEAD_OUT_VOX * HF.pitch
DEAD_IN = DEAD_IN_VOX * HF.pitch
print(f"[p1h] hull pitch={HF.pitch:.4f} dead_out={DEAD_OUT:.4f} "
      f"dead_in={DEAD_IN:.4f}", flush=True)

_BARY = torch.tensor([
    [1/3, 1/3, 1/3],
    [1/2, 1/2, 0.0], [0.0, 1/2, 1/2], [1/2, 0.0, 1/2],
    [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3],
], dtype=torch.float32, device=DEVICE)

def hull_loss(verts_t, faces_l):
    tri = verts_t[faces_l]
    pts = torch.einsum("sk,fkc->fsc", _BARY, tri).reshape(-1, 3)
    do = HF.dist(pts).view(-1, _BARY.shape[0])
    di = HF.dist_in(pts).view(-1, _BARY.shape[0])
    pen = F.relu(do - DEAD_OUT) + (W_IN / W_OUT) * F.relu(di - DEAD_IN)
    area = 0.5 * torch.cross(tri[:, 1] - tri[:, 0],
                             tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    w = area.detach() / (area.detach().sum() + 1e-12)
    return (pen.mean(1) * w).sum(), do, di

def iou_fn(vv, ff):
    vt = torch.tensor(np.asarray(vv), dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(ff, np.int32), dtype=torch.int32, device=DEVICE)
    return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
gtd_t = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(NV)]
gtfg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(NV)]
gtdf_t = [torch.from_numpy(gtdiff[i]).float().to(DEVICE) for i in range(NV)]

ho0 = heldout_exam(ctx, V, Fa)
print(f"[p1h] BASE train={iou_fn(V, Fa):.4f} ho16={ho0[0]:.4f} "
      f"hair={ho0[1]}", flush=True)
t0 = time.time()

for rnd in range(WRAP_ROUNDS):
    last = rnd == WRAP_ROUNDS - 1
    verts_t = torch.tensor(V, dtype=torch.float32,
                           device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(Fa.astype(np.int32), dtype=torch.int32,
                           device=DEVICE)
    faces_l = faces_t.long()
    src, dst, deg, excl = build_adj(Fa.astype(np.int32), len(V))
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS,
                                                       eta_min=LR_MIN)
    for step in range(STEPS):
        opt.zero_grad()
        me = mean_edge_of(verts_t.detach(), src, dst)
        sl = dl = fl = torch.tensor(0.0, device=DEVICE)
        for i in range(NV):
            sil, ndc_z, fg, diff = render_sdd(ctx, verts_t, faces_t,
                                              mvps[i], views[i])
            sl = sl + F.l1_loss(sil[0], targets[i])
            dl = dl + depth_loss_masked(ndc_z, fg, gtd_t[i], gtfg_t[i])
            fl = fl + F.l1_loss(diff, gtdf_t[i])
        sl, dl, fl = sl / NV, dl / NV, fl / NV
        wh = W_OUT * min(1.0, (step + 1) / RAMP)
        if last and ANNEAL > 0 and step >= STEPS - ANNEAL:
            wh = wh * max(0.1, (STEPS - step) / ANNEAL)
        hl, do, di = hull_loss(verts_t, faces_l)
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
            print(f"[p1h r{rnd}] step {step+1}/{STEPS} sil={sl.item():.4f} "
                  f"hull={hl.item():.5f} deep_in="
                  f"{(di.mean(1) > DEAD_IN).sum().item()}f "
                  f"({time.time()-t0:.0f}s)", flush=True)
    V = verts_t.detach().cpu().numpy().astype(np.float64)
    ho_r = heldout_exam(ctx, V, Fa)
    print(f"[p1h r{rnd}] ho16={ho_r[0]:.4f} hair={ho_r[1]} V={len(V)}",
          flush=True)
    if last: break
    # adaptive DLFL subdivision where surface is deep inside the hull
    with torch.no_grad():
        _, do, di = hull_loss(verts_t, faces_l)
        deep = di.mean(1) > SUB_THR_VOX * HF.pitch
    fids = torch.where(deep)[0].cpu().numpy()
    if len(fids) == 0:
        print(f"[p1h r{rnd}] no deep faces, stop refining", flush=True)
        continue
    if len(fids) > SUB_CAP:
        depths = di.mean(1)[deep].cpu().numpy()
        fids = fids[np.argsort(depths)[::-1][:SUB_CAP]]
    V, Fa, ne = dlfl_subdivide_arrays(V, Fa, fids.tolist())
    ok, _ = check_watertight(Fa)
    print(f"[p1h r{rnd}] subdivided {ne} edges at {len(fids)} deep faces "
          f"-> V={len(V)} F={len(Fa)} watertight={ok}", flush=True)
    assert ok

hof = heldout_exam(ctx, V, Fa)
print(f"\n[p1h] FINAL train={iou_fn(V, Fa):.4f} ho16={hof[0]:.4f} "
      f"hair={hof[1]} maxblob={hof[2]} V={len(V)} F={len(Fa)}")
print(f"[p1h] delta ho16: {ho0[0]:.4f} -> {hof[0]:.4f} "
      f"({(hof[0]-ho0[0])*100:+.2f}pts)", flush=True)
np.savez_compressed(os.path.join(OUTD, f"cow_{SHAPE}_{TAG}.npz"),
                    verts=V, tris=Fa)
with open(os.path.join(OUTD, f"cow_{SHAPE}_{TAG}.obj"), "w") as fh:
    for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
    for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
print(f"[p1h] saved cow_{SHAPE}_{TAG}.npz/.obj", flush=True)
