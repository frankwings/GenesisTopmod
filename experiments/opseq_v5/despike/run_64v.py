"""Our v22+L_qual pipeline under DMesh's EXACT mv-recon setup:
- 64 star cameras (8 azimuths x 8 elevations -70..70 step 20), object fills frame
  (radius = 2 x mesh max-radius, fov = 2*atan(0.5) = 53.13 deg) — mirrors
  make_star_cameras(8,8, distance=2.0) in DMesh's normalized domain.
- Supervision: silhouette L1 + masked depth + Lambertian diffuse (camera-space
  headlight [0,0,-1], per-vertex normals) — DMesh uses diffuse+depth.
- Same C2F recipe, v22 escape-seeded surgery, L_qual=0.01 (Phase 0 winner).
Exam unchanged: 16 held-out views.

SHAPE env selects mesh. Run via bg_task (approx 10x cost of 6-view run).
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import math, time, collections
import numpy as np, torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, label as cc_label
import nvdiffrast.torch as dr

import cow_v13
from cow_v13 import (midpoint_subdivide, build_pairs, fold_loss, build_adj,
                     centroid, spike_pen, sliver_pen, tube_mask, mean_edge_of,
                     DEVICE, OUT, WARMUP_STEPS, LAP_WARMUP_STEPS,
                     DEPTH_WARMUP_STEPS, W_LAP_BOOST, W_FOLD, FOLD_START,
                     W_SPIKE, W_SLIVER, W_TUBE, TUBE_EVERY)
from eval_local_refine import (setup_scene, render_views_n, compute_iou_n,
                               depth_loss_masked, laplacian_loss,
                               edge_length_loss, load_obj, normalize_to_range,
                               BUNNY_PATH, IMG_RES, LR, LR_MIN,
                               W_DEPTH, W_LAP, W_EDGE)
from eval_extrude_v3 import orbit_cameras, render_sil_and_depth
from pipeline.cameras import transform_to_clip, look_at
from surgery_lib import surgery, _propagate_flag
from escape_util import escape_mask

import viz_snap
_SQRT3_4 = 4.0 * (3.0 ** 0.5)


def _qual_loss(verts_t, faces_t):
    tri = verts_t[faces_t.long()]
    e0 = tri[:, 1] - tri[:, 0]
    e1 = tri[:, 2] - tri[:, 1]
    e2 = tri[:, 0] - tri[:, 2]
    l2 = (e0 * e0).sum(-1) + (e1 * e1).sum(-1) + (e2 * e2).sum(-1)
    area = 0.5 * torch.cross(e0, -e2, dim=-1).norm(dim=-1)
    q = _SQRT3_4 * area / (l2 + 1e-12)
    return (1.0 - q).mean()

SHAPE = os.environ.get("SHAPE", "armadillo")
TAG = os.environ.get("TAG", f"{SHAPE}_64v")
W_QUAL = float(os.environ.get("W_QUAL", "0.01"))
W_DIFF = float(os.environ.get("W_DIFF", "1.0"))
DILATE = int(os.environ.get("DILATE", "2"))
cow_v13.TUBE_THR = float(os.environ.get("TUBE_THR", "0.4"))
NV = 64
TRAIN_RES = int(os.environ.get("TRAIN_RES", str(IMG_RES)))   # supervision resolution (GT + training renders); exam stays IMG_RES
RENDER_BATCH = int(os.environ.get("RENDER_BATCH", "1"))       # 0 = per-view loop (legacy), 1 = batched nvdiffrast
print(f"[run_64v] SHAPE={SHAPE} TAG={TAG} W_QUAL={W_QUAL} W_DIFF={W_DIFF} RENDER_BATCH={RENDER_BATCH}", flush=True)


def star_cameras(max_radius, device=DEVICE):
    """DMesh make_star_cameras(8,8,distance=2) equivalent in our frame."""
    R = 2.0 * max_radius
    fov = math.degrees(2.0 * math.atan(0.5))
    mvps, views = [], []
    elevs = [-70 + 20 * k for k in range(8)]      # theta = k*pi/9 - pi/2, k=1..8
    azs = [45.0 * a for a in range(8)]
    for el in elevs:
        m = orbit_cameras(n=8, elevation_deg=float(el), radius=R,
                          fov_deg=fov, azimuths_deg=azs, device=device)
        if isinstance(m, tuple): m = m[0]
        mvps.append(m)
        for az in azs:
            er, ar = math.radians(el), math.radians(az)
            eye = np.array([R * math.cos(er) * math.cos(ar),
                            R * math.sin(er),
                            R * math.cos(er) * math.sin(ar)])
            views.append(look_at(tuple(eye), center=(0, 0, 0), up=(0, 1, 0),
                                 device=device))
    return torch.cat(mvps, 0), torch.stack(views, 0)  # [64,4,4] mvp, view


def vertex_normals(verts_t, faces_l, nv):
    tri = verts_t[faces_l]
    fn = torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1)
    vn = torch.zeros(nv, 3, device=verts_t.device, dtype=verts_t.dtype)
    for k in range(3):
        vn = vn.index_add(0, faces_l[:, k], fn)
    return vn / (vn.norm(dim=-1, keepdim=True) + 1e-12)


def render_sdd(ctx, verts_t, faces_t, mvp, view):
    """silhouette, ndc_z, fg, diffuse (camera-space headlight)."""
    H = W = TRAIN_RES
    pos_clip = transform_to_clip(verts_t, mvp)
    rast, _ = dr.rasterize(ctx, pos_clip, faces_t, resolution=[H, W])
    faces_l = faces_t.long()
    ones = torch.ones(1, verts_t.shape[0], 3, dtype=torch.float32,
                      device=verts_t.device)
    color, _ = dr.interpolate(ones, rast, faces_t)
    sil = dr.antialias(color, rast, pos_clip, faces_t)[..., :1]
    fg = rast[0, :, :, 3] > 0
    clip_zw = pos_clip[0, :, 2:4]
    zw, _ = dr.interpolate(clip_zw.unsqueeze(0).contiguous(), rast, faces_t)
    ndc_z = zw[0, :, :, 0] / zw[0, :, :, 1].clamp(min=1e-6)
    vn = vertex_normals(verts_t, faces_l, verts_t.shape[0])
    vn_cam = vn @ view[:3, :3].T                     # world -> camera rotation
    shade = vn_cam[:, 2].abs().unsqueeze(0).unsqueeze(-1)  # |n_cam . (0,0,-1)|
    diff_img, _ = dr.interpolate(shade.contiguous(), rast, faces_t)
    diff = diff_img[0, :, :, 0] * fg.float()
    return sil, ndc_z, fg, diff


def render_sdd_batch(ctx, verts_t, faces_t, mvps, views):
    """Batched render of all N views in one nvdiffrast call.

    mvps  : [N,4,4]  model-view-projection matrices (stacked)
    views : [N,4,4]  view (world->camera) matrices (stacked)

    Returns sil [N,H,W], ndc_z [N,H,W], fg [N,H,W] bool, diff [N,H,W].
    Semantics match render_sdd per view (sil has the leading batch dim squeezed
    from antialias [N,H,W,1] -> [N,H,W]).
    """
    H = W = TRAIN_RES
    N     = mvps.shape[0]
    V_num = verts_t.shape[0]

    # ── Clip positions [N,V,4]: one MVP multiply for all views ────────────────
    ones_V  = torch.ones(V_num, 1, dtype=verts_t.dtype, device=verts_t.device)
    verts_h = torch.cat([verts_t, ones_V], dim=-1)          # [V,4]
    pos     = (mvps @ verts_h.T).transpose(1, 2).contiguous()  # [N,V,4]

    # ── Rasterize: one call for all N views ───────────────────────────────────
    rast, _ = dr.rasterize(ctx, pos, faces_t, resolution=[H, W])  # [N,H,W,4]

    # ── Silhouette ────────────────────────────────────────────────────────────
    # nvdiffrast requires attr to be [N,V,C] when rast is batched
    ones_attr = torch.ones(N, V_num, 3, dtype=torch.float32, device=verts_t.device)
    color, _  = dr.interpolate(ones_attr, rast, faces_t)           # [N,H,W,3]
    sil       = dr.antialias(color, rast, pos, faces_t)[..., :1]   # [N,H,W,1]

    # ── Foreground mask ───────────────────────────────────────────────────────
    fg = rast[:, :, :, 3] > 0                                      # [N,H,W]

    # ── Depth (NDC z) ─────────────────────────────────────────────────────────
    clip_zw  = pos[..., 2:4].contiguous()                          # [N,V,2]
    zw, _    = dr.interpolate(clip_zw, rast, faces_t)             # [N,H,W,2]
    ndc_z    = zw[..., 0] / zw[..., 1].clamp(min=1e-6)            # [N,H,W]

    # ── Diffuse (camera-space headlight) ──────────────────────────────────────
    # vertex normals computed once for all views
    vn       = vertex_normals(verts_t, faces_t.long(), V_num)      # [V,3]
    # vn_cam[n,v,i] = sum_j views[n,i,j] * vn[v,j]  (= vn @ R_n.T per view)
    vn_cam   = torch.einsum('nij,vj->nvi', views[:, :3, :3], vn)  # [N,V,3]
    shade    = vn_cam[..., 2].abs().contiguous()                   # [N,V]
    diff_img, _ = dr.interpolate(shade.unsqueeze(-1).contiguous(), rast, faces_t)  # [N,H,W,1]
    diff     = diff_img[..., 0] * fg.float()                       # [N,H,W]

    return sil[..., 0], ndc_z, fg, diff


def render_normals(ctx, verts_t, faces_t, mvp, view):
    """camera-space normal image (H,W,3), zero outside fg. Palfinger-style supervision."""
    H = W = TRAIN_RES
    pos_clip = transform_to_clip(verts_t, mvp)
    rast, _ = dr.rasterize(ctx, pos_clip, faces_t, resolution=[H, W])
    faces_l = faces_t.long()
    vn = vertex_normals(verts_t, faces_l, verts_t.shape[0])
    vn_cam = (vn @ view[:3, :3].T).unsqueeze(0).contiguous()
    nimg, _ = dr.interpolate(vn_cam, rast, faces_t)
    fg = (rast[0, :, :, 3] > 0).float().unsqueeze(-1)
    return nimg[0] * fg


def make_gt_normals(ctx, mvps, views, shape):
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{shape}.obj"))
    gv = normalize_to_range(gv)
    vt = torch.tensor(gv, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(gf, dtype=torch.int32, device=DEVICE)
    with torch.no_grad():
        return [render_normals(ctx, vt, ft, mvps[i], views[i]) for i in range(NV)]


def make_gt(ctx, mvps, views, shape):
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{shape}.obj"))
    gv = normalize_to_range(gv)
    max_r = float(np.linalg.norm(gv, axis=1).max())
    vt = torch.tensor(gv, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(gf, dtype=torch.int32, device=DEVICE)
    sils, deps, diffs = [], [], []
    with torch.no_grad():
        for i in range(NV):
            sil, z, fg, diff = render_sdd(ctx, vt, ft, mvps[i], views[i])
            sils.append(((1.0 - sil[0, :, :, 0].cpu().numpy()) * 255)
                        .clip(0, 255).astype(np.uint8))
            deps.append(z.cpu().numpy())
            diffs.append(diff.cpu().numpy())
    return (np.stack(sils), np.stack(deps).astype(np.float32),
            np.stack(diffs).astype(np.float32), max_r)


def optimize_phase64(ctx, verts_np, tris_np, gt, gtd, gtdiff, mvps, views,
                     steps, label, settle=False, use_fold=False, use_tube=False):
    targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
    gtd_t   = [torch.from_numpy(gtd[i]).float().to(DEVICE) for i in range(NV)]
    gtfg_t  = [torch.from_numpy(gt[i] < 128).to(DEVICE)   for i in range(NV)]
    gtdf_t  = [torch.from_numpy(gtdiff[i]).float().to(DEVICE) for i in range(NV)]
    # Stacked GT tensors for batched loss (lists kept for face_residual / exam)
    if RENDER_BATCH:
        from batch_losses import sil_loss_batch, depth_loss_batch, diff_loss_batch
        _gtd_stack  = torch.stack(gtd_t)                       # [NV,H,W]
        _gtfg_stack = torch.stack(gtfg_t)                      # [NV,H,W]
        _gtdf_stack = torch.stack(gtdf_t)                      # [NV,H,W]
        _tgt_batch  = targets[..., 0]                          # [NV,H,W]
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
            tmask = cow_v13.tube_mask(verts_t.detach(), excl, me)
        opt.zero_grad()
        if RENDER_BATCH:
            # ── Batched 64-view render (single rasterize call) ────────────────
            sil_b, ndc_b, fg_b, diff_b = render_sdd_batch(ctx, verts_t, faces_t, mvps, views)
            sl = sil_loss_batch(sil_b, _tgt_batch)
            dl = depth_loss_batch(ndc_b, fg_b, _gtd_stack, _gtfg_stack)
            fl = diff_loss_batch(diff_b, _gtdf_stack)
        else:
            # ── Legacy per-view loop (RENDER_BATCH=0) ────────────────────────
            sl = dl = fl = torch.tensor(0.0, device=DEVICE)
            for i in range(NV):
                sil, ndc_z, fg, diff = render_sdd(ctx, verts_t, faces_t, mvps[i], views[i])
                sl = sl + F.l1_loss(sil[0], targets[i])
                dl = dl + depth_loss_masked(ndc_z, fg, gtd_t[i], gtfg_t[i])
                fl = fl + F.l1_loss(diff, gtdf_t[i])
            sl, dl, fl = sl / NV, dl / NV, fl / NV
        loss = (sl + dw * dl + W_DIFF * fl
                + w_lap * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t)
                + W_QUAL * _qual_loss(verts_t, faces_t)
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
        if viz_snap.enabled():
            viz_snap.snap(ctx, mvps, verts_t.detach().cpu().numpy(), tris_np, f"Stage 1 [{label}] step {step+1}/{steps}", step=step)
        if step % 200 == 0:
            iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
            print(f"  [{label}] {step:4d}/{steps} iou={iou:.4f} "
                  f"sil={float(sl):.4f} diff={float(fl):.4f}", flush=True)
    iou = compute_iou_n(render_views_n(ctx, verts_t, faces_t, mvps), gt)
    print(f"  [{label}] END iou={iou:.4f}", flush=True)
    return verts_t.detach().cpu().numpy().astype(np.float64), iou


C2F_SUBDIV = os.environ.get("C2F_SUBDIV", "cc")   # DEFAULT cc (TopMod Catmull-Clark + triangulate_all; quads kept through Stage 1) since 2026-09-11 (LESSONS 23: = midpoint) | midpoint (numpy 1->4, legacy)

def _c2f(v, t, polys):
    """Coarse-to-fine refinement between the cc levels. Returns (v, tris, polys-or-None)."""
    if C2F_SUBDIV == "cc":
        import cc_subdiv
        v, polys, t = cc_subdiv.cc_subdivide(v, polys if polys is not None else np.asarray(t).tolist())
        print(f"[run_64v] TopMod Catmull-Clark: V={len(v)} polys={len(polys)} tris={len(t)}", flush=True)
        return v, t, polys
    v, t = midpoint_subdivide(v, t); return v, t, None

_CUR_ADJ = None
_MVPS = None
_GT = None
_orig_tube = cow_v13.tube_mask


def _set_faces(Fa):
    global _CUR_ADJ
    adj = collections.defaultdict(set)
    for a, b, c in np.asarray(Fa):
        a, b, c = int(a), int(b), int(c)
        adj[a] |= {b, c}; adj[b] |= {a, c}; adj[c] |= {a, b}
    _CUR_ADJ = adj


@torch.no_grad()
def _tube_gated(v, excl, mean_edge):
    thin = _orig_tube(v, excl, mean_edge)
    if _MVPS is None or _CUR_ADJ is None:
        return thin
    esc = escape_mask(v, _MVPS, _GT, dilate=DILATE)
    keep = _propagate_flag(thin.cpu().numpy(), esc.cpu().numpy(), _CUR_ADJ)
    return torch.from_numpy(keep).to(v.device)


cow_v13.tube_mask = _tube_gated


def heldout_exam(ctx, v, t):
    gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
    gv = normalize_to_range(gv)
    gvt = torch.tensor(gv, dtype=torch.float32, device=DEVICE)
    gft = torch.tensor(gf, dtype=torch.int32, device=DEVICE)
    azs = [22.5 + 22.5 * i for i in range(16)]
    mv = orbit_cameras(n=16, elevation_deg=20.0, radius=2.5, azimuths_deg=azs, device=DEVICE)
    if isinstance(mv, tuple): mv = mv[0]
    pvt = torch.tensor(v, dtype=torch.float32, device=DEVICE)
    pft = torch.tensor(np.asarray(t, np.int32), dtype=torch.int32, device=DEVICE)
    gs = render_views_n(ctx, gvt, gft, mv)
    ps = render_views_n(ctx, pvt, pft, mv)
    inter = un = hair = 0; maxblob = 0
    for i in range(16):
        g = gs[i] > 0.5; p = ps[i] > 0.5
        inter += int((g & p).sum()); un += int((g | p).sum())
        out = p & ~binary_dilation(g, iterations=2); hair += int(out.sum())
        lab, nb = cc_label(out)
        for k in range(1, nb + 1):
            maxblob = max(maxblob, int((lab == k).sum()))
    return inter / max(un, 1), hair, maxblob


HULL_INIT = int(os.environ.get("HULL_INIT", "0"))          # 1: project the init icosphere onto the 64-view voting hull before Stage 1 DR
HULL_INIT_ITERS = int(os.environ.get("HULL_INIT_ITERS", "12"))
HULL_INIT_SMOOTH = float(os.environ.get("HULL_INIT_SMOOTH", "0.5"))   # uniform-Laplacian blend per iteration (keeps triangles from folding on concave hulls)
S1_STEPS = int(os.environ.get("S1_STEPS", "800"))            # DR steps per Stage-1 level (cc2, cc3)


def hull_init_project(ctx, mvps, gv, gf, v, t, HF=None):
    """Soft projection of the init mesh onto the voting hull surface: iterate
    (move each vertex by a fraction of its signed hull distance along the field gradient,
    then blend with the uniform Laplacian). Uses only the 64 training silhouettes."""
    if HF is None:
        from hull_field import build_vote_hull
        gvn = normalize_to_range(gv)
        HF = build_vote_hull(ctx, mvps, gvn, gf, np.asarray(v), DEVICE, nres=256, hires=512, vote=2)
    tt = np.asarray(t, np.int64); n = len(v)
    nb = [[] for _ in range(n)]
    for a, b, c in tt: nb[a] += [b, c]; nb[b] += [a, c]; nb[c] += [a, b]
    nb = [np.unique(x) for x in nb]
    pts = torch.tensor(np.asarray(v), dtype=torch.float32, device=DEVICE)
    for it in range(HULL_INIT_ITERS):
        p = pts.clone().requires_grad_(True)
        sd = HF.dist(p) - HF.dist_in(p)                       # signed distance to hull surface (+ outside)
        g, = torch.autograd.grad(sd.sum(), p)
        gn = g / g.norm(dim=1, keepdim=True).clamp_min(1e-9)
        with torch.no_grad():
            step = (0.6 * sd).clamp(-3 * HF.pitch, 3 * HF.pitch)      # bounded step: never jump across a limb in one go
            pts = pts - step[:, None] * gn
            if HULL_INIT_SMOOTH > 0:
                P = pts.cpu().numpy().astype(np.float64)
                L = np.stack([P[ix].mean(0) for ix in nb])
                pts = torch.tensor((1 - HULL_INIT_SMOOTH) * P + HULL_INIT_SMOOTH * L, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            r = (HF.dist(pts) - HF.dist_in(pts)).abs()
        print(f"[run_64v] hull_init it{it}: |sd| mean={r.mean().item()/HF.pitch:.2f} max={r.max().item()/HF.pitch:.2f} vox", flush=True)
    return pts.cpu().numpy().astype(np.float64)


def main():
    global _MVPS, _GT, W_DIFF, W_DEPTH
    _SEED = int(os.environ.get("SEED", "0")); torch.manual_seed(_SEED); np.random.seed(_SEED)
    ctx = dr.RasterizeCudaContext()
    _real_scene = None
    REAL_DATA = os.environ.get("REAL_DATA", "")
    if REAL_DATA:
        from real_scene import load_real_scene
        _real_scene = load_real_scene(REAL_DATA, DEVICE)
        mvps, views = _real_scene.mvps, _real_scene.views
        gt, gtd, gtdiff, max_r = _real_scene.gt, _real_scene.gtd, _real_scene.gtdiff, _real_scene.max_r
        _MVPS, _GT = mvps, gt
        W_DIFF = 0.0; W_DEPTH = 0.0
        import phase1b_pipeline as _p1b; _p1b.heldout_exam = _real_scene.heldout_exam
        print(f"[run_64v] REAL_DATA={REAL_DATA}: forced W_DEPTH=0 W_DIFF=0", flush=True)
    else:
        # bootstrap: need max_radius before cameras -> load GT once
        gv, _gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
        max_r = float(np.linalg.norm(normalize_to_range(gv), axis=1).max())
        mvps, views = star_cameras(max_r)
        gt, gtd, gtdiff, _ = make_gt(ctx, mvps, views, SHAPE)
        _MVPS, _GT = mvps, gt
    print(f"[run_64v] max_r={max_r:.3f} R={2*max_r:.3f} gt shapes "
          f"{gt.shape} {gtd.shape} {gtdiff.shape}", flush=True)

    if REAL_DATA:
        import cc_subdiv; v, polys, t = cc_subdiv.icosphere_cc2()
        print(f"[run_64v] REAL_DATA CC icosphere: V={len(v)} quads={len(polys)} tris={len(t)}", flush=True)
    else:
        scene = setup_scene(SHAPE, DEVICE)  # only for init icosphere
        v, t = scene["init_verts"], scene["init_tris"]; polys = None
        if C2F_SUBDIV == "cc":
            import cc_subdiv; v, polys, t = cc_subdiv.icosphere_cc2()
            print(f"[run_64v] C2F_SUBDIV=cc: TopMod icosphere cc2 V={len(v)} quads={len(polys)} tris={len(t)}", flush=True)
    viz_snap.snap(ctx, mvps, v, t, "Stage 1 [init icosphere]", hold=30)
    if HULL_INIT:
        if _real_scene:
            _HF = _real_scene.hull(ctx, extra_pts=np.asarray(v))
            v = hull_init_project(ctx, mvps, None, None, v, t, HF=_HF)
        else:
            v = hull_init_project(ctx, mvps, gv, _gf, v, t)
        viz_snap.snap(ctx, mvps, v, t, "Stage 1 [hull-projected init]", hold=30)

    def iou_fn(vv, ff):
        vt = torch.tensor(np.asarray(vv), dtype=torch.float32, device=DEVICE)
        ft = torch.tensor(np.asarray(ff, dtype=np.int32), dtype=torch.int32, device=DEVICE)
        return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

    def escape_fn(Vnp):
        vt = torch.tensor(np.asarray(Vnp), dtype=torch.float32, device=DEVICE)
        return escape_mask(vt, mvps, gt, dilate=DILATE).cpu().numpy()

    t0 = time.time()
    if os.environ.get("RESUME_FROM"):
        # early-hole experiment: mesh already went through cc2/cc3 (+ handles); continue with the cc4 phase
        z = np.load(os.environ["RESUME_FROM"]); v, t = z["verts"].astype(np.float64), z["tris"].astype(np.int64); polys = None
        print(f"[run_64v] RESUME_FROM {os.environ['RESUME_FROM']}: V={len(v)} F={len(t)}", flush=True)
    else:
        _set_faces(t)
        v, _ = optimize_phase64(ctx, v, t, gt, gtd, gtdiff, mvps, views, S1_STEPS, "cc2",
                                use_fold=True, use_tube=True)
        v, t, polys = _c2f(v, t, polys); _set_faces(t)
        v, _ = optimize_phase64(ctx, v, t, gt, gtd, gtdiff, mvps, views, S1_STEPS, "cc3",
                                settle=True, use_tube=True)
    _heldout_exam = _real_scene.heldout_exam if _real_scene else heldout_exam
    if os.environ.get("STOP_AFTER") == "cc3":
        # early-hole experiment: hand the cc3 mesh (1.9k faces) to the handle stage before any further subdivision
        t = np.asarray(t, np.int32)
        np.savez(f"{OUT}/cow_{TAG}.npz", verts=np.asarray(v, np.float64), tris=t)
        ho, hair, mb = _heldout_exam(ctx, v, t)
        print(f"[run_64v] STOP_AFTER=cc3: saved cow_{TAG}.npz V={len(v)} F={len(t)} ho16={ho:.4f} hair={hair}", flush=True)
        return
    v, t, polys = _c2f(v, t, polys); _set_faces(t)
    v, iou_train = optimize_phase64(ctx, v, t, gt, gtd, gtdiff, mvps, views, 800,
                                    "cc4", settle=True, use_fold=True, use_tube=True)
    print(f"[train] iou={iou_train:.4f} V={len(v)} ({time.time()-t0:.0f}s)", flush=True)
    v, t = surgery(np.asarray(v, np.float64), np.asarray(t, np.int64),
                   iou_fn=iou_fn, iou_budget=3e-4, global_cap=1.5e-3,
                   max_rounds=8, max_grow=4, escape_fn=escape_fn)
    _set_faces(t)
    viz_snap.snap(ctx, mvps, v, t, "Stage 1 [despike surgery 1: DLFL collapse of escaped needles]", hold=30)
    v, _ = optimize_phase64(ctx, v, np.asarray(t, np.int32), gt, gtd, gtdiff,
                            mvps, views, 400, "settle", settle=True,
                            use_fold=True, use_tube=True)
    v, t = surgery(np.asarray(v, np.float64), np.asarray(t, np.int64),
                   iou_fn=iou_fn, iou_budget=2e-4, global_cap=6e-4,
                   max_rounds=4, max_grow=4, escape_fn=escape_fn)
    viz_snap.snap(ctx, mvps, v, t, "Stage 1 [despike surgery 2] -> 7k faces", hold=30)
    iou_final = iou_fn(v, t); t = np.asarray(t, np.int32)
    with open(f"{OUT}/cow_{TAG}.obj", "w") as fh:
        for x, y, z in v: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in t: fh.write(f"f {a+1} {b+1} {c+1}\n")
    np.savez(f"{OUT}/cow_{TAG}.npz", verts=v, tris=t)
    ho, hair, mb = _heldout_exam(ctx, v, t)
    print(f"\n=== {TAG.upper()} RESULT ===")
    print(f"train64 IoU={iou_final:.4f} (post-train {iou_train:.4f})")
    print(f"heldout16: IoU={ho:.4f} hair_px={hair} maxblob={mb}")
    print(f"V={len(v)} time={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
