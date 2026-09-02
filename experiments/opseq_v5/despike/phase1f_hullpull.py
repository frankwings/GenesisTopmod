"""Phase 1f: space carving as a differentiable loss (hull attraction).

Boss's insight (2026-09-01): a red pixel in ANY view proves no material on that
ray. The 64-training-view silhouette intersection = GT visual hull. Any vertex
sitting outside the hull is PROVABLY misplaced -- pull it to the nearest hull
voxel. For webbing between fingers the nearest hull material is the adjacent
finger, so both webbing skins get pulled laterally onto the fingers: the slab
collapses without any topology change (manifold trivially preserved).

This is the mid-slab fix that per-op carving could not do: silhouette gradients
cannot see through-slab redness (back surface covers the pixel), but the hull
distance field sees it in 3D directly.

Supervision honesty: hull is built ONLY from the 64 training silhouettes
(diag_spacecarve.npz). Exam = 16 held-out views, untouched.

Run: TAG=p1f64 STEPS=800 W_HULL=20 python3 despike/phase1f_hullpull.py
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import time
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
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

SHAPE = os.environ.get("SHAPE", "armadillo")
TAG = os.environ.get("TAG", "p1f64")
BASE_NPZ = os.environ.get("BASE_NPZ", "/tmp/liou_cow_viz/cow_armadillo_p1d64c.npz")
HULL_NPZ = os.environ.get("HULL_NPZ", "/tmp/liou_cow_viz/diag_spacecarve.npz")
STEPS = int(os.environ.get("STEPS", "800"))
W_HULL = float(os.environ.get("W_HULL", "20.0"))
W_QUAL = float(os.environ.get("W_QUAL", "0.01"))
W_DIFF = float(os.environ.get("W_DIFF", "1.0"))
RAMP = int(os.environ.get("RAMP", "200"))
# Phase 1f-c knobs
HULL_MODE = os.environ.get("HULL_MODE", "npz")   # npz | vote
NRES_V = int(os.environ.get("NRES_V", "256"))    # vote-hull voxel res
HIRES = int(os.environ.get("HIRES", "512"))      # vote-hull silhouette res
VOTE = int(os.environ.get("VOTE", "2"))          # views required to prove "outside"
ANNEAL = int(os.environ.get("ANNEAL", "200"))    # final steps: hull weight decay
DEAD_VOX = float(os.environ.get("DEAD_VOX", "1.0"))  # dead zone in voxel units
OUTD = "/tmp/liou_cow_viz"

torch.manual_seed(0); np.random.seed(0)
ctx = dr.RasterizeCudaContext()
gv, _gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
max_r = float(np.linalg.norm(normalize_to_range(gv), axis=1).max())
mvps, views = star_cameras(max_r)
gt, gtd, gtdiff, _ = make_gt(ctx, mvps, views, SHAPE)
run_64v._MVPS, run_64v._GT = mvps, gt
cow_v13.N_VIEWS = 64
p1b._MVPS, p1b._GT = mvps, gt
p1b.SHAPE = SHAPE

z = np.load(BASE_NPZ)
V0, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ok, _ = check_watertight(Fa)
print(f"[p1f] base {BASE_NPZ} V={len(V0)} F={len(Fa)} watertight={ok} "
      f"W_HULL={W_HULL} STEPS={STEPS}", flush=True)

# ---------------------------------------------------------------- hull field
def build_vote_hull():
    """Phase 1f-c hull: robust to subpixel-thin structures.

    Old hull (diag_spacecarve) ate 3.05% of GT material (up to 19 voxels deep
    at the hand) because a finger thinner than one pixel vanishes from a
    single antialiased silhouette and the 64-view intersection deletes it.
    Fixes: (1) hard-coverage silhouettes at HIRES=512 (any raster hit =
    inside), (2) 1px dilation of inside, (3) VOTING -- a voxel is "outside"
    only if >=VOTE views say so; no single-view veto.
    """
    from pipeline.cameras import transform_to_clip
    gvn = normalize_to_range(gv)
    gvt = torch.tensor(gvn, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(_gf, dtype=torch.int32, device=DEVICE)
    lo_ = np.minimum(gvn.min(0), V0.min(0)) - 0.02
    hi_ = np.maximum(gvn.max(0), V0.max(0)) + 0.02
    fgs = []
    with torch.no_grad():
        for i in range(NV):
            pos = transform_to_clip(gvt, mvps[i])
            rast, _ = dr.rasterize(ctx, pos, gft, resolution=[HIRES, HIRES])
            fg = (rast[0, :, :, 3] > 0).float()
            fg = (F.max_pool2d(fg[None, None], 3, 1, 1)[0, 0] > 0)  # 1px dilate
            fgs.append(fg)
        axes = [torch.linspace(float(lo_[a]), float(hi_[a]), NRES_V,
                               device=DEVICE) for a in range(3)]
        gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
        P = torch.stack([gx, gy, gz], -1).view(-1, 3)
        votes = torch.zeros(P.shape[0], dtype=torch.uint8, device=DEVICE)
        ones_c = torch.ones(P.shape[0], 1, device=DEVICE)
        Ph = torch.cat([P, ones_c], 1)
        for i in range(NV):
            clip = (mvps[i] @ Ph.T).T
            w = clip[:, 3].clamp(min=1e-8)
            x, y = clip[:, 0] / w, clip[:, 1] / w
            u = ((x + 1) * 0.5 * HIRES).long().clamp(0, HIRES - 1)
            v = ((y + 1) * 0.5 * HIRES).long().clamp(0, HIRES - 1)
            inb = (x.abs() <= 1) & (y.abs() <= 1)
            outside = inb & ~fgs[i][v, u]
            votes += outside.to(torch.uint8)
        hull_ = (votes < VOTE).view(NRES_V, NRES_V, NRES_V).cpu().numpy()
    return lo_, hi_, NRES_V, hull_

if HULL_MODE == "vote":
    lo, hi, NRES, gt_hull = build_vote_hull()
    print(f"[p1f] vote-hull NRES={NRES} HIRES={HIRES} VOTE={VOTE} "
          f"hull_vox={gt_hull.sum()}", flush=True)
else:
    h = np.load(HULL_NPZ)
    lo, hi, NRES = h["lo"], h["hi"], int(h["nres"])
    gt_hull = h["gt_hull"]
sp = (hi - lo) / (NRES - 1)                       # per-axis pitch
dist = distance_transform_edt(~gt_hull, sampling=tuple(sp)).astype(np.float32)
print(f"[p1f] hull field: NRES={NRES} max_dist={dist.max():.4f} "
      f"outside_frac={(dist > 0).mean():.3f}", flush=True)
# volume [1,1,D=X,H=Y,W=Z]; grid last dim = (w=Z, h=Y, d=X)
vol = torch.from_numpy(dist).unsqueeze(0).unsqueeze(0).to(DEVICE)
lo_t = torch.tensor(lo, dtype=torch.float32, device=DEVICE)
hi_t = torch.tensor(hi, dtype=torch.float32, device=DEVICE)

def hull_dist(verts_t):
    g = 2.0 * (verts_t - lo_t) / (hi_t - lo_t) - 1.0          # [V,3] in x,y,z
    grid = g[:, [2, 1, 0]].view(1, 1, 1, -1, 3)               # (z,y,x) order
    d = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border",
                      align_corners=True)
    return d.view(-1)                                          # [V] world units

# Phase 1f-b: face-INTERIOR sampling. Vertex-only pull left "tent faces"
# spanning finger gaps: all 3 verts legally on fingers, face interior deep
# outside hull (p1f64: 1/3494 verts out but 364 centroids + 1342 edge
# midpoints out, max_d=0.064). Sample barycentric interior points so the
# face sheet itself receives hull gradient; area-weight so stretched tent
# faces pull hardest.
_BARY = torch.tensor([
    [1/3, 1/3, 1/3],
    [1/2, 1/2, 0.0], [0.0, 1/2, 1/2], [1/2, 0.0, 1/2],
    [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3],
], dtype=torch.float32, device=DEVICE)                         # [S,3]

DEAD_W = DEAD_VOX * float(np.max(sp))   # 1f-c dead zone: free within 1 voxel
                                        # (kills EDT stair-step roughness)

def hull_surface_loss(verts_t, faces_l):
    tri = verts_t[faces_l]                                     # [F,3,3]
    pts = torch.einsum("sk,fkc->fsc", _BARY, tri).reshape(-1, 3)  # [F*S,3]
    d = hull_dist(pts).view(-1, _BARY.shape[0])                # [F,S]
    pen = F.relu(d - DEAD_W)
    area = 0.5 * torch.cross(tri[:, 1] - tri[:, 0],
                             tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    w = area.detach() / (area.detach().sum() + 1e-12)
    return (pen.mean(1) * w).sum(), d

# ---------------------------------------------------------------- baseline
def iou_fn(vv, ff):
    vt = torch.tensor(np.asarray(vv), dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(ff, np.int32), dtype=torch.int32, device=DEVICE)
    return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

ho0 = heldout_exam(ctx, V0, Fa)
d0 = hull_dist(torch.tensor(V0, dtype=torch.float32, device=DEVICE))
print(f"[p1f] BASE train={iou_fn(V0, Fa):.4f} ho16={ho0[0]:.4f} hair={ho0[1]} "
      f"| verts outside hull: {(d0 > 1e-4).sum().item()}/{len(V0)} "
      f"mean_d={d0.mean().item():.5f} max_d={d0.max().item():.4f}", flush=True)

# ---------------------------------------------------------------- optimize
targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
gtd_t = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(NV)]
gtfg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(NV)]
gtdf_t = [torch.from_numpy(gtdiff[i]).float().to(DEVICE) for i in range(NV)]
verts_t = torch.tensor(V0, dtype=torch.float32, device=DEVICE).requires_grad_(True)
faces_t = torch.tensor(Fa, dtype=torch.int32, device=DEVICE)
faces_l = faces_t.long()
src, dst, deg, excl = build_adj(Fa.astype(np.int32), len(V0))
opt = torch.optim.Adam([verts_t], lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR_MIN)

t0 = time.time()
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
    wh = W_HULL * min(1.0, (step + 1) / RAMP)
    if ANNEAL > 0 and step >= STEPS - ANNEAL:   # 1f-c: anneal, let sil polish
        frac = (STEPS - step) / ANNEAL
        wh = wh * max(0.1, frac)
    hl, hd_s = hull_surface_loss(verts_t, faces_l)
    hd = hull_dist(verts_t)
    hl = hl + F.relu(hd - DEAD_W).mean()        # vertex term, dead-zoned
    loss = (sl + W_DEPTH * dl + W_DIFF * fl
            + W_LAP * laplacian_loss(verts_t, faces_t)
            + W_EDGE * edge_length_loss(verts_t, faces_t)
            + W_QUAL * _qual_loss(verts_t, faces_t)
            + W_SPIKE * spike_pen(verts_t, src, dst, deg, me)
            + W_SLIVER * sliver_pen(verts_t, faces_l, me)
            + wh * hl)
    loss.backward()
    opt.step(); sched.step()
    if (step + 1) % 100 == 0 or step == 0:
        with torch.no_grad():
            out_n = (hd > 1e-4).sum().item()
            out_s = (hd_s > 1e-4).sum().item()
        print(f"[p1f] step {step+1}/{STEPS} sil={sl.item():.4f} "
              f"hull={hl.item():.5f} out_v={out_n} out_samp={out_s} "
              f"wh={wh:.1f} ({time.time()-t0:.0f}s)", flush=True)

Vf = verts_t.detach().cpu().numpy().astype(np.float64)
hof = heldout_exam(ctx, Vf, Fa)
df = hull_dist(verts_t.detach())
print(f"\n[p1f] FINAL train={iou_fn(Vf, Fa):.4f} ho16={hof[0]:.4f} "
      f"hair={hof[1]} maxblob={hof[2]}")
print(f"[p1f] delta ho16: {ho0[0]:.4f} -> {hof[0]:.4f} ({(hof[0]-ho0[0])*100:+.2f}pts)")
print(f"[p1f] outside-hull verts: {(d0 > 1e-4).sum().item()} -> "
      f"{(df > 1e-4).sum().item()}  mean_d {d0.mean():.5f} -> {df.mean():.5f}",
      flush=True)
out = os.path.join(OUTD, f"cow_{SHAPE}_{TAG}.npz")
np.savez_compressed(out, verts=Vf, tris=Fa)
print(f"[p1f] saved {out}", flush=True)
