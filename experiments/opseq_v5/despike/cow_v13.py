"""V13: v10 sanctioned recipe (6-view-only, online spike+sliver+fold priors)
+ NEW online thin-tube prior:
  every TUBE_EVERY steps recompute tube mask (vertex whose nearest NON-2-hop
  vertex is closer than 0.4*mean_edge => part of a degenerate needle/tube),
  then penalize masked verts' distance to neighbor centroid (collapse pull).
This targets exactly the chain-needle that evades per-vertex laplacian.
All supervision = the original 6 training views. Held-out 16 views = exam only.
"""
import sys, os, time
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label as cc_label
from eval_local_refine import (
    setup_scene, render_sil_and_depth, render_views_n, compute_iou_n,
    depth_loss_masked, laplacian_loss, edge_length_loss,
    IMG_RES, N_VIEWS, LR, LR_MIN, W_DEPTH, W_LAP, W_EDGE,
)
from eval_extrude_v3 import orbit_cameras

DEVICE = "cuda"
OUT = "/tmp/liou_cow_viz"
WARMUP_STEPS = 20
LAP_WARMUP_STEPS = 50
DEPTH_WARMUP_STEPS = 80
W_LAP_BOOST = 0.40
W_FOLD = 0.02
FOLD_START = 400
W_SPIKE = 400.0
SPIKE_THR = 0.85
W_SLIVER = 2000.0
W_TUBE = 100.0
TUBE_EVERY = 50
TUBE_THR = 0.4


def midpoint_subdivide(verts, tris):
    verts = list(map(tuple, verts))
    em = {}
    def mid(a, b):
        k = (a, b) if a < b else (b, a)
        if k not in em:
            va, vb = verts[a], verts[b]
            verts.append(((va[0]+vb[0])/2, (va[1]+vb[1])/2, (va[2]+vb[2])/2))
            em[k] = len(verts) - 1
        return em[k]
    nt = []
    for a, b, c in tris:
        ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
        nt += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
    return np.array(verts, dtype=np.float64), np.array(nt, dtype=np.int32)


def build_pairs(tris_np):
    e2f = {}
    for fi, (a, b, c) in enumerate(tris_np):
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            e2f.setdefault(k, []).append(fi)
    return np.array([p for p in e2f.values() if len(p) == 2], dtype=np.int64)


def fold_loss(verts_t, faces_l, pairs_t):
    tri = verts_t[faces_l]
    n = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    n = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    d = (n[pairs_t[:, 0]] * n[pairs_t[:, 1]]).sum(-1)
    return F.relu(-d).pow(2).mean()


def build_adj(tris_np, nv):
    src, dst = [], []
    es = set()
    for a, b, c in tris_np:
        for e in ((a, b), (b, c), (c, a)):
            k = (min(e), max(e))
            if k in es: continue
            es.add(k)
            src += [e[0], e[1]]; dst += [e[1], e[0]]
    src = torch.tensor(src, dtype=torch.long, device=DEVICE)
    dst = torch.tensor(dst, dtype=torch.long, device=DEVICE)
    deg = torch.zeros(nv, device=DEVICE).index_add_(
        0, src, torch.ones_like(src, dtype=torch.float32)).clamp(min=1).unsqueeze(-1)
    # 2-hop exclusion mask (V x V bool): self + 1-hop + 2-hop
    A = torch.zeros(nv, nv, dtype=torch.bool, device=DEVICE)
    A[src, dst] = True
    A2 = (A.float() @ A.float()) > 0
    excl = A | A2 | torch.eye(nv, dtype=torch.bool, device=DEVICE)
    return src, dst, deg, excl


def centroid(v, src, dst, deg):
    cen = torch.zeros_like(v).index_add_(0, src, v[dst])
    return cen / deg


def spike_pen(v, src, dst, deg, mean_edge):
    lap = (v - centroid(v, src, dst, deg)).norm(dim=-1)
    return F.relu(lap - SPIKE_THR * mean_edge).pow(2).mean()


def sliver_pen(v, faces_l, mean_e):
    tri = v[faces_l]
    e0 = (tri[:, 1] - tri[:, 0]).norm(dim=-1)
    e1 = (tri[:, 2] - tri[:, 1]).norm(dim=-1)
    e2 = (tri[:, 0] - tri[:, 2]).norm(dim=-1)
    lmax = torch.stack([e0, e1, e2], -1).max(-1).values
    area = 0.5 * torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0],
                             dim=-1).norm(dim=-1)
    h = 2 * area / (lmax + 1e-12)
    p = F.relu(0.25 * mean_e - h).pow(2).mean()
    for e in (e0, e1, e2):
        p = p + F.relu(e - 3 * mean_e).pow(2).mean()
    return p


@torch.no_grad()
def tube_mask(v, excl, mean_edge):
    D = torch.cdist(v, v)
    D[excl] = 1e9
    return D.min(1).values < TUBE_THR * mean_edge


def mean_edge_of(v, src, dst):
    return (v[src] - v[dst]).norm(dim=-1).mean()


def optimize_phase(ctx, verts_np, tris_np, gt, gtd, mvps, steps, label,
                   settle=False, use_fold=False, use_tube=False):
    targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gt_depth_t = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(N_VIEWS)]
    gt_fg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(N_VIEWS)]
    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=DEVICE)
    faces_l = faces_t.long()
    src, dst, deg, excl = build_adj(tris_np, len(verts_np))
    pairs_t = (torch.tensor(build_pairs(tris_np), device=DEVICE)
               if use_fold else None)
    opt = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps, eta_min=LR_MIN)
    warmup = WARMUP_STEPS if settle else 0
    lap_boost = LAP_WARMUP_STEPS if settle else 0
    depth_warm = DEPTH_WARMUP_STEPS if settle else 0
    tmask = None
    for step in range(steps):
        if warmup > 0:
            frac = 1.0 - warmup / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = LR * max(frac, 0.05)
            warmup -= 1
        w_lap = W_LAP_BOOST if lap_boost > 0 else W_LAP
        dw = W_DEPTH * (1.0 - depth_warm / DEPTH_WARMUP_STEPS) if depth_warm > 0 else W_DEPTH
        if lap_boost > 0: lap_boost -= 1
        if depth_warm > 0: depth_warm -= 1
        me = mean_edge_of(verts_t.detach(), src, dst)
        if use_tube and step % TUBE_EVERY == 0:
            tmask = tube_mask(verts_t.detach(), excl, me)
        opt.zero_grad()
        sl = torch.tensor(0.0, device=DEVICE)
        dl = torch.tensor(0.0, device=DEVICE)
        for i in range(N_VIEWS):
            sil, ndc_z, fg = render_sil_and_depth(ctx, verts_t, faces_t, mvps[i],
                                                  (IMG_RES, IMG_RES))
            sl += F.l1_loss(sil[0], targets[i])
            dl += depth_loss_masked(ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        sl /= N_VIEWS; dl /= N_VIEWS
        loss = (sl + dw * dl + w_lap * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t)
                + W_SPIKE * spike_pen(verts_t, src, dst, deg, me)
                + W_SLIVER * sliver_pen(verts_t, faces_l, me))
        if pairs_t is not None and step >= FOLD_START:
            ramp = min(1.0, (step - FOLD_START) / max(1, (steps - FOLD_START) * 0.5))
            loss = loss + W_FOLD * ramp * fold_loss(verts_t, faces_l, pairs_t)
        if tmask is not None and tmask.any():
            cen = centroid(verts_t, src, dst, deg)
            loss = loss + W_TUBE * (verts_t[tmask] - cen[tmask].detach()
                                    ).norm(dim=-1).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()
        if step % 200 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
            nt = int(tmask.sum()) if tmask is not None else -1
            print(f"  [{label}] {step:4d}/{steps} iou={iou:.4f} tube_v={nt}",
                  flush=True)
    iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
    v_out = verts_t.detach()
    tm = tube_mask(v_out, excl, mean_edge_of(v_out, src, dst))
    print(f"  [{label}] END iou={iou:.4f} final_tube_v={int(tm.sum())}", flush=True)
    return v_out.cpu().numpy().astype(np.float64), iou


def heldout_exam(ctx, v, t, scene):
    """16 held-out views: render GT mesh and pred, compare. Exam only."""
    from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
    gt_v, gt_f = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), "cow.obj"))
    gt_v = normalize_to_range(gt_v)
    gvt = torch.tensor(gt_v, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(gt_f, dtype=torch.int32, device=DEVICE)
    azs = [22.5 + 22.5 * i for i in range(16)]
    mv = orbit_cameras(n=16, elevation_deg=20.0, radius=2.5,
                       azimuths_deg=azs, device=DEVICE)
    pvt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    pft = torch.tensor(t, dtype=torch.int32, device=DEVICE)
    gt_sils = render_views_n(ctx, gvt, gft, mv)
    pr_sils = render_views_n(ctx, pvt, pft, mv)
    inter = un = hair = 0; maxblob = 0
    for i in range(16):
        g = gt_sils[i] > 0.5; p = pr_sils[i] > 0.5
        inter += int((g & p).sum()); un += int((g | p).sum())
        out = p & ~binary_dilation(g, iterations=2)
        hair += int(out.sum())
        lab, nb = cc_label(out)
        for k in range(1, nb + 1):
            maxblob = max(maxblob, int((lab == k).sum()))
    return inter / max(un, 1), hair, maxblob


def main():
    torch.manual_seed(0); np.random.seed(0)
    scene = setup_scene("cow", DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]
    t0 = time.time()
    v, t = scene["init_verts"], scene["init_tris"]
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc2")
    v, t = midpoint_subdivide(v, t)
    v, _ = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc3",
                          settle=True, use_tube=True)
    v, t = midpoint_subdivide(v, t)
    v, iou = optimize_phase(ctx, v, t, gt, gtd, mvps, 800, "cc4",
                            settle=True, use_fold=True, use_tube=True)
    with open(f"{OUT}/cow_v13.obj", "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")
    print(f"saved {OUT}/cow_v13.obj", flush=True)
    ho_iou, ho_hair, ho_mb = heldout_exam(ctx, v, t, scene)
    print(f"\n=== V13 RESULT (tube prior in-training) ===")
    print(f"train6 IoU={iou:.4f}")
    print(f"heldout16: IoU={ho_iou:.4f} hair_px={ho_hair} maxblob={ho_mb}")
    print(f"refs: v10 train6=0.9827 ho=0.8714/32222/2009")
    print(f"time: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
