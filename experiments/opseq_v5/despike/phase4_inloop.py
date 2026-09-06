"""Phase 4: DLFL flip sweep INSIDE the optimizer (every FLIP_EVERY steps).

Alternating untangle<->refit (3e) showed the continuous optimizer re-creates
tangles (6v 11%->19.5% SI, 64v 25%->38% per refit). Fix at the source: run a
DLFL edge-flip sweep (+1 tangential smoothing iteration) every N steps while
optimizing, rebuilding adjacency in place. Vertex count never changes, so
Adam state stays valid; topology changes are DLFL-only (manifold preserved).

MODE=6v : sil+depth on 6 views + soup distance field (TARGET_OBJ)
MODE=64v: sil+depth+diffuse on 64 views + voting hull field (1f-c)
Both: lap/edge/qual/spike/sliver/fold regularizers.
Targets: self-intersecting faces < 5% with no IoU loss.

Run: MODE=6v TAG=p4_6 BASE_NPZ=... TARGET_OBJ=... python3 despike/phase4_inloop.py
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
import nvdiffrast.torch as dr

import cow_v13
from cow_v13 import (build_adj, build_pairs, fold_loss, mean_edge_of, spike_pen,
                     sliver_pen, DEVICE, W_SPIKE, W_SLIVER, W_FOLD)
from eval_local_refine import (setup_scene, render_views_n, compute_iou_n,
                               load_obj, normalize_to_range, BUNNY_PATH,
                               depth_loss_masked, laplacian_loss, edge_length_loss,
                               LR, LR_MIN, W_DEPTH, W_LAP, W_EDGE)
from eval_extrude_v3 import render_sil_and_depth
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight
from eval_dmesh import load_any_mesh
from dlfl_untangle import (flip_sweep, tangential_smooth, si_faces, fold_frac,
                           collapse_short_edges, vertex_dihedral)

SHAPE = os.environ.get("SHAPE", "armadillo")
MODE = os.environ.get("MODE", "6v")
TAG = os.environ.get("TAG", f"p4_{MODE}")
BASE_NPZ = os.environ["BASE_NPZ"]
TARGET_OBJ = os.environ.get("TARGET_OBJ", "")
STEPS = int(os.environ.get("STEPS", "600"))
FLIP_EVERY = int(os.environ.get("FLIP_EVERY", "25"))
SMOOTH_ITERS = int(os.environ.get("SMOOTH_ITERS", "1"))
SMOOTH_LAM = float(os.environ.get("SMOOTH_LAM", "0.2"))
W_T = float(os.environ.get("W_T", "20.0"))
W_QUAL = float(os.environ.get("W_QUAL", "0.01"))
W_DIFF = float(os.environ.get("W_DIFF", "1.0"))
W_NORMAL = float(os.environ.get("W_NORMAL", "0.0"))   # L1 on camera-space normal image (Palfinger-style); 0 = off
FOLD_MULT = float(os.environ.get("FOLD_MULT", "1.0"))
COLLAPSE_EVERY = int(os.environ.get("COLLAPSE_EVERY", "0"))   # 0 = off
COLLAPSE_RATIO = float(os.environ.get("COLLAPSE_RATIO", "0.3"))
ADAPT_REMESH = int(os.environ.get("ADAPT_REMESH", "0"))   # curvature-adaptive remeshing on DLFL ops (split where curved, collapse where flat)
ADAPT_LO = float(os.environ.get("ADAPT_LO", "8.0"))        # dihedral (deg) at/below which a vertex counts as flat
ADAPT_HI = float(os.environ.get("ADAPT_HI", "30.0"))       # dihedral at/above which a vertex counts as fully curved
ADAPT_TMIN = float(os.environ.get("ADAPT_TMIN", "0.5"))    # target edge length (x me0) in curved regions
ADAPT_TMAX = float(os.environ.get("ADAPT_TMAX", "1.6"))    # target edge length (x me0) in flat regions
ADAPT_CRATIO = float(os.environ.get("ADAPT_CRATIO", "0.8"))   # collapse edges shorter than CRATIO x local target (Botsch-Kobbelt 4/5)
ADAPT_SRATIO = float(os.environ.get("ADAPT_SRATIO", "1.3333")) # split faces whose longest edge exceeds SRATIO x local target (4/3): midpoint split then lands in [2/3, 1]*L, not [1/2, 1]*L
ADAPT_SPLIT_FRAC = float(os.environ.get("ADAPT_SPLIT_FRAC", "0.05"))
ADAPT_COLLAPSE_FRAC = float(os.environ.get("ADAPT_COLLAPSE_FRAC", "0.2"))  # collapse cap per pass (x F): ~4x the split cap in vertex terms so a pass can be net-negative (B-K needs collapse to keep up); uncapped pure-Python DLFL was 100% CPU for 25 min with no pass done  # cap: faces split per pass as a fraction of F
ADAPT_MAX_F = int(os.environ.get("ADAPT_MAX_F", "60000"))   # stop splitting above this face count (256^2 supervision ceiling; pure-Python DLFL cost)
ADAPT_FOLD = float(os.environ.get("ADAPT_FOLD", "70.0"))    # dihedral above this = tangle/fold, not a feature: never split, let collapse clean it
ADAPT_SMOOTH_K = int(os.environ.get("ADAPT_SMOOTH_K", "3"))  # measure curvature on a Taubin-smoothed copy: real curvature survives, SI jitter does not
ADAPT_SI_GATE = float(os.environ.get("ADAPT_SI_GATE", "0.05"))  # no splits while the self-intersecting face fraction exceeds this (clean before subdividing)
ADAPT_LMIN_PX = float(os.environ.get("ADAPT_LMIN_PX", "2.0"))  # floor on target edge length in PIXELS of the training images: below ~2 px the loss cannot see an edge, refinement is pure cost
ADAPT_MODE = os.environ.get("ADAPT_MODE", "curvature")       # "curvature" | "residual": what decides the target edge length
ADAPT_CURV_W = float(os.environ.get("ADAPT_CURV_W", "0.0"))    # residual mode: blend in curvature drive, t = max(t_err, W * t_curv)
ADAPT_E_LO = float(os.environ.get("ADAPT_E_LO", "0.5"))        # residual quantiles (data-adaptive): below E_LO-quantile = fitted (long target)
ADAPT_E_HI = float(os.environ.get("ADAPT_E_HI", "0.9"))        # above E_HI-quantile = badly fitted (short target)
ADAPT_R_SPLIT = float(os.environ.get("ADAPT_R_SPLIT", "0.05"))   # residual mode (absolute): mean image error per covered pixel-view above this = unfit -> short target
ADAPT_R_KEEP = float(os.environ.get("ADAPT_R_KEEP", "0.01"))     # below this AND flat -> may coarsen; in between -> keep current local edge length (never coarsen a region the fit still needs)
ADAPT_ABS = int(os.environ.get("ADAPT_ABS", "1"))                # 1 = absolute pixel-scale thresholds (shape-agnostic), 0 = quantiles (v7: coarsened fertility to 2.4k V)
ADAPT_KEEP_FLAT = int(os.environ.get("ADAPT_KEEP_FLAT", "1"))  # 1: coarsen only where fitted AND flat; 0: coarsen wherever fitted (a fitted thin limb does not need its density either)
ADAPT_EPS_PX = float(os.environ.get("ADAPT_EPS_PX", "0.5"))     # dunyach: approximation tolerance epsilon in pixels -> L = sqrt(6 eps/k - 3 eps^2)
ADAPT_GRADE = float(os.environ.get("ADAPT_GRADE", "0.0"))       # sizing-field gradation (Lipschitz alpha): L(v) <= L(u) + alpha*|uv|; 0 = off (Dunyach / attention-flow analogue)
ADAPT_NU_GAIN = float(os.environ.get("ADAPT_NU_GAIN", "0.2"))   # velocity mode (Palfinger): ref_len *= 1 + (nu/nu_med - 1) * gain
ADAPT_VEL_GUARD = float(os.environ.get("ADAPT_VEL_GUARD", "0.0"))  # Palfinger-style guard for ANY mode: faces whose vertices moved > K x median displacement since the last remesh are still moving/jittering -> not split this pass (0 = off)
# ADAPT_MODE: curvature (dihedral proxy) | dunyach (principal curvature + eps) | velocity (Palfinger closed loop)
#             | curv_uniform (3DV-2026: high-curvature split + uniform split by mean edge) | residual (image residual, ours)

def principal_kmax(Vx, Fx):
    """max |principal curvature| per vertex from cotangent mean curvature H (Meyer 2003) and
    angle-deficit Gaussian curvature K: kmax = |H| + sqrt(max(H^2 - K, 0)). Barycentric area."""
    Vx = np.asarray(Vx, float); Fx = np.asarray(Fx, np.int64); nv = len(Vx)
    A = np.zeros(nv); KH = np.zeros((nv, 3)); ang_sum = np.zeros(nv)
    fa = 0.5 * np.linalg.norm(np.cross(Vx[Fx[:, 1]] - Vx[Fx[:, 0]], Vx[Fx[:, 2]] - Vx[Fx[:, 0]]), axis=1)
    for k in range(3): np.add.at(A, Fx[:, k], fa / 3.0)
    for k in range(3):
        i0, i1, i2 = Fx[:, k], Fx[:, (k + 1) % 3], Fx[:, (k + 2) % 3]
        e1 = Vx[i1] - Vx[i0]; e2 = Vx[i2] - Vx[i0]
        cosang = (e1 * e2).sum(1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1) + 1e-12)
        ang = np.arccos(np.clip(cosang, -1, 1)); np.add.at(ang_sum, i0, ang)
        # cot at vertex i0 weights edge (i1,i2): contributes to KH of i1 and i2
        cot = cosang / (np.sqrt(np.clip(1 - cosang ** 2, 1e-12, None)))
        d = Vx[i1] - Vx[i2]
        np.add.at(KH, i1, cot[:, None] * d); np.add.at(KH, i2, -cot[:, None] * d)
    A = np.maximum(A, 1e-12)
    H = np.linalg.norm(KH, axis=1) / (4.0 * A)             # |mean curvature normal| / 2 -> H
    Kg = (2 * np.pi - ang_sum) / A
    disc = np.maximum(H * H - Kg, 0.0)
    return H + np.sqrt(disc)

def grade_sizing(Vx, Fx, L, alpha, passes=8):
    """Lipschitz gradation: L(v) <= L(u) + alpha*|uv| over edges (Dunyach 2013; attention-flow analogue)."""
    src = np.concatenate([Fx[:, 0], Fx[:, 1], Fx[:, 2], Fx[:, 1], Fx[:, 2], Fx[:, 0]])
    dst = np.concatenate([Fx[:, 1], Fx[:, 2], Fx[:, 0], Fx[:, 0], Fx[:, 1], Fx[:, 2]])
    el = np.linalg.norm(Vx[src] - Vx[dst], axis=1); L = L.copy()
    for _ in range(passes):
        cand = L[dst] + alpha * el
        np.minimum.at(L, src, cand)
    return L

def local_edge_len(Vx, Fx):
    src = np.concatenate([Fx[:, 0], Fx[:, 1], Fx[:, 2], Fx[:, 1], Fx[:, 2], Fx[:, 0]])
    dst = np.concatenate([Fx[:, 1], Fx[:, 2], Fx[:, 0], Fx[:, 0], Fx[:, 1], Fx[:, 2]])
    le = np.zeros(len(Vx)); c = np.zeros(len(Vx))
    np.add.at(le, src, np.linalg.norm(Vx[src] - Vx[dst], axis=1)); np.add.at(c, src, 1)
    return le / np.maximum(c, 1)

def face_residual(Vx, Fx):
    """Per-face image-fit error summed over the training views: |rendered sil - GT sil| + masked
    |depth - GT depth| on pixels the face covers (nvdiffrast triangle-id buffer), plus GT-foreground
    pixels we fail to cover ("missing"), attributed to the nearest rendered face. This is where the
    loss says the fit is bad -- refinement goes there, not to whatever is merely thin/curved."""
    from pipeline.cameras import transform_to_clip
    from scipy.ndimage import distance_transform_edt
    vt = torch.tensor(Vx, dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(Fx.astype(np.int32), dtype=torch.int32, device=DEVICE)
    err = np.zeros(len(Fx)); cov = np.zeros(len(Fx))
    with torch.no_grad():
        for i in range(NV):
            pos = transform_to_clip(vt, mvps[i])
            H = W = int(targets.shape[1])
            rast, _ = dr.rasterize(ctx, pos, ft, resolution=[H, W])
            tid = rast[0, :, :, 3].long().cpu().numpy() - 1          # -1 = background
            fg = tid >= 0
            ones = torch.ones(1, len(Vx), 3, dtype=torch.float32, device=DEVICE)
            col, _ = dr.interpolate(ones, rast, ft)
            sil = dr.antialias(col, rast, pos, ft)[0, :, :, 0].cpu().numpy()
            zw, _ = dr.interpolate(pos[0, :, 2:4].unsqueeze(0).contiguous(), rast, ft)
            ndc = (zw[0, :, :, 0] / zw[0, :, :, 1].clamp(min=1e-6)).cpu().numpy()
            gsil = targets[i, :, :, 0].cpu().numpy(); gfg = gtfg_t[i].cpu().numpy(); gd = gtd_t[i].cpu().numpy()
            e = np.abs(sil - gsil)                                     # silhouette disagreement
            both = fg & gfg
            e[both] += np.abs(ndc[both] - gd[both]) * W_DEPTH          # depth residual, same weight as the loss
            # covered pixels -> their face
            np.add.at(err, tid[fg], e[fg]); np.add.at(cov, tid[fg], 1.0)
            # missing pixels (GT fg, we render bg) -> nearest rendered face
            miss = gfg & ~fg
            if miss.any() and fg.any():
                _, idx = distance_transform_edt(~fg, return_indices=True)
                near = tid[idx[0][miss], idx[1][miss]]
                np.add.at(err, near, e[miss] + 1.0); np.add.at(cov, near, 1.0)
    return err, cov

def _taubin_np(Vx, Fx, iters, lam=0.5, mu=-0.53):
    Vx = np.asarray(Vx, float).copy(); nv = len(Vx)
    src = np.concatenate([Fx[:, 0], Fx[:, 1], Fx[:, 2], Fx[:, 1], Fx[:, 2], Fx[:, 0]])
    dst = np.concatenate([Fx[:, 1], Fx[:, 2], Fx[:, 0], Fx[:, 0], Fx[:, 1], Fx[:, 2]])
    deg = np.bincount(src, minlength=nv).astype(float)[:, None]
    for _ in range(iters):
        for k in (lam, mu):
            cen = np.zeros_like(Vx); np.add.at(cen, src, Vx[dst]); cen /= np.maximum(deg, 1)
            Vx += k * (cen - Vx)
    return Vx

def _si_ring_mask(Vx, Fx):
    """faces that self-intersect, plus every face sharing a vertex with one (1-ring)."""
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vx), o3d.utility.Vector3iVector(Fx.astype(np.int32)))
    prs = np.asarray(om.get_self_intersecting_triangles())
    bad = np.zeros(len(Fx), bool)
    if len(prs):
        bad[np.unique(prs)] = True
        badv = np.zeros(len(Vx), bool); badv[np.unique(Fx[bad])] = True
        bad |= badv[Fx].any(1)
    return bad, (len(np.unique(prs)) / len(Fx) if len(prs) else 0.0)
COLLAPSE_MAX = int(os.environ.get("COLLAPSE_MAX", "300"))
COLLAPSE_FRAC = float(os.environ.get("COLLAPSE_FRAC", "0"))
COLLAPSE_ABS = int(os.environ.get("COLLAPSE_ABS", "1"))     # threshold = ratio x INITIAL mean edge (fixed), not the current mean: stops the runaway (3holes final stage ate 24% of V)  # if >0: per-call cap = frac x current face count (small meshes were eaten by a fixed cap: fertility cc3 1.9k -> 378 faces)
SI_PUSH = float(os.environ.get("SI_PUSH", "0.0"))   # nudge intersecting pairs apart (x mean edge)
SI_EVERY = int(os.environ.get("SI_EVERY", "25"))    # Open3D self-intersection check cadence (5.6 s at 36k faces = the real CPU hog, not DLFL)
SUBDIV_ALL = int(os.environ.get("SUBDIV_ALL", "0"))
SUBDIV_TOP = int(os.environ.get("SUBDIV_TOP", "0"))
SNAPSHOT_EVERY = int(os.environ.get("SNAPSHOT_EVERY", "0"))  # render front/back/back-closeup frames every N steps (video)
SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", f"/tmp/liou_cow_viz/frames_{TAG}")
SNAPSHOT_MODE = os.environ.get("SNAPSHOT_MODE", "3")  # "3" = front/back/closeup, "64" = all training cameras mosaic  # Phase 6: DLFL-subdivide only the N largest faces (resolution equalization)  # Phase 5: global DLFL midpoint subdivision passes before optimizing
LAP_MULT = float(os.environ.get("LAP_MULT", "1.0"))  # Phase 5: fairing strength (back smoothness)
OUTD = "/tmp/liou_cow_viz"
os.makedirs(OUTD, exist_ok=True)
torch.manual_seed(0); np.random.seed(0)

_SQRT3_4 = 4.0 * (3.0 ** 0.5)
def _qual_loss(verts_t, faces_t):
    tri = verts_t[faces_t.long()]
    e0 = tri[:, 1] - tri[:, 0]; e1 = tri[:, 2] - tri[:, 1]; e2 = tri[:, 0] - tri[:, 2]
    l2 = (e0 * e0).sum(-1) + (e1 * e1).sum(-1) + (e2 * e2).sum(-1)
    area = 0.5 * torch.cross(e0, -e2, dim=-1).norm(dim=-1)
    return (1.0 - _SQRT3_4 * area / (l2 + 1e-12)).mean()

# ---------------------------------------------------------------- scene
z = np.load(BASE_NPZ)
V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
if SUBDIV_ALL > 0:
    # resolution is the common denominator of "fingers not grown" and "back
    # not smooth": mean edge 0.13 vs finger width ~0.1-0.2. Global DLFL
    # subdivide_edge on every edge + stellate (all TopMod ops, watertight).
    from phase1c_pipeline import dlfl_subdivide_arrays
    for _ in range(SUBDIV_ALL):
        V, Fa, ne = dlfl_subdivide_arrays(V, Fa, list(range(len(Fa))))
        wt, _ = check_watertight(Fa); assert wt
        print(f"[p4] global DLFL subdivision: split {ne} edges -> V={len(V)} F={len(Fa)}", flush=True)
    if os.environ.get("SNAPSHOT_DIR") and SNAPSHOT_EVERY > 0:
        pass  # (frame written at step 0 by the loop; scene not built yet here)
if SUBDIV_TOP > 0:
    # partial subdivision: the N largest faces (+1-ring, DLFL subdivide_edge + stellate).
    from phase1c_pipeline import dlfl_subdivide_arrays
    area = 0.5 * np.linalg.norm(np.cross(V[Fa[:, 1]] - V[Fa[:, 0]], V[Fa[:, 2]] - V[Fa[:, 0]]), axis=1)
    fids = np.argsort(-area)[:SUBDIV_TOP].tolist()
    V, Fa, ne = dlfl_subdivide_arrays(V, Fa, fids)
    wt, _ = check_watertight(Fa); assert wt
    print(f"[p4] partial DLFL subdivision of {SUBDIV_TOP} largest faces: split {ne} edges -> V={len(V)} F={len(Fa)}", flush=True)
if MODE == "64v":
    import run_64v
    from run_64v import render_sdd
    from hull_field import build_vote_hull
    ctx = dr.RasterizeCudaContext()
    gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
    gvn = normalize_to_range(gv)
    mvps, views = run_64v.star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
    PX_SIZE = 2.0 * float(np.linalg.norm(gvn, axis=1).max()) / 256.0   # image half-height = max_radius (fov 2*atan(0.5), R = 2*max_radius)
    gt, gtd, gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
    cow_v13.N_VIEWS = 64
    HF = build_vote_hull(ctx, mvps, gvn, gf_gt, V, DEVICE, nres=256, hires=512, vote=2)
    DEAD = 1.0 * HF.pitch
    gtdf_t = [torch.from_numpy(gtdiff[i]).float().to(DEVICE) for i in range(64)]
    gtn_t = run_64v.make_gt_normals(ctx, mvps, views, SHAPE) if float(os.environ.get("W_NORMAL", "0")) > 0 else None
    def field_dist(pts): return F.relu(HF.dist(pts) - DEAD)
else:
    scene = setup_scene(SHAPE, DEVICE)
    ctx, mvps = scene["ctx"], scene["mvps"]
    gt, gtd = scene["gt_uint8"], scene["gt_depths"]
if MODE == "6v" and not TARGET_OBJ:
    # DMesh-free 6v: voting hull from the 6 TRAINING silhouettes only
    # (vote=1: with 6 clean views any single view proves "outside";
    # 1024px 2x-supersampled coverage keeps thin parts).
    from hull_field import build_vote_hull
    gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
    gvn = normalize_to_range(gv)
    HF = build_vote_hull(ctx, mvps, gvn, gf_gt, V, DEVICE, nres=256, hires=1024,
                         vote=1, ss_thr=0.25)
    DEAD = 1.0 * HF.pitch
    print(f"[p4] 6v hull field: vox={HF.hull.sum()} pitch={HF.pitch:.4f}", flush=True)
    def field_dist(pts): return F.relu(HF.dist(pts) - DEAD)
elif MODE == "6v":
    tv, tf = load_any_mesh(TARGET_OBJ)
    tv = np.asarray(tv, np.float32); tf = np.asarray(tf, np.uint32)
    NRES = 256
    lo = np.minimum(tv.min(0), V.min(0)) - 0.03; hi = np.maximum(tv.max(0), V.max(0)) + 0.03
    sp = (hi - lo) / (NRES - 1); PITCH = float(sp.max()); DEAD = 0.5 * PITCH
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
    def field_dist(pts):
        g = 2.0 * (pts - lo_t) / (hi_t - lo_t) - 1.0
        grid = g[:, [2, 1, 0]].view(1, 1, 1, -1, 3)
        d = F.grid_sample(vol, grid, mode="bilinear", padding_mode="border",
                          align_corners=True).view(-1)
        return F.relu(d - DEAD)
NV = len(mvps)
p1b._MVPS, p1b._GT = mvps, gt
p1b.SHAPE = SHAPE
targets = torch.from_numpy((gt < 128).astype(np.float32)).unsqueeze(-1).to(DEVICE)
gtd_t = [torch.from_numpy(np.asarray(gtd[i], np.float32)).to(DEVICE) for i in range(NV)]
gtfg_t = [torch.from_numpy(gt[i] < 128).to(DEVICE) for i in range(NV)]

_BARY = torch.tensor([[1/3, 1/3, 1/3], [1/2, 1/2, 0.0], [0.0, 1/2, 1/2], [1/2, 0.0, 1/2],
                      [2/3, 1/6, 1/6], [1/6, 2/3, 1/6], [1/6, 1/6, 2/3]],
                     dtype=torch.float32, device=DEVICE)

def field_loss(verts_t, faces_l):
    tri = verts_t[faces_l]
    pts = torch.einsum("sk,fkc->fsc", _BARY, tri).reshape(-1, 3)
    pen = field_dist(pts).view(-1, _BARY.shape[0]).mean(1)
    area = 0.5 * torch.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0], dim=-1).norm(dim=-1)
    w = area.detach() / (area.detach().sum() + 1e-12)
    return (pen * w).sum() + field_dist(verts_t).mean()

def _snapshot(step, Vn, Fa):
    from PIL import Image, ImageDraw, ImageFont
    from viz_render import render as _vr, ROWS as _ROWS
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    if SNAPSHOT_MODE == "64":
        if os.environ.get("SNAPSHOT_DIR"):
            import viz_snap
            return viz_snap.snap(ctx, mvps, Vn, Fa, f"{os.environ.get('SNAPSHOT_TITLE', TAG)} step {step+1}/{STEPS}", step=step)
        return _snapshot64(step, Vn, Fa)
    views = [r for r in _ROWS if r[0] in ("front", "back", "back closeup")]
    res = 512
    canvas = Image.new("L", (res * len(views), res + 40), 255); d = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception: font = ImageFont.load_default()
    for j, (nm, kw) in enumerate(views):
        canvas.paste(Image.fromarray(_vr(Vn, Fa, res=res, **kw)), (j * res, 40))
    d.text((10, 8), f"{TAG}  step {step:4d}/{STEPS}   V={len(Vn)} F={len(Fa)}", fill=0, font=font)
    canvas.save(f"{SNAPSHOT_DIR}/frame_{step:05d}.png")

def _snapshot64(step, Vn, Fa, res=192, cols=8):
    """One frame = mosaic of flat-shaded renders from ALL training cameras (8x8 for 64v)."""
    from PIL import Image, ImageDraw, ImageFont
    vt = torch.tensor(Vn, dtype=torch.float32, device=DEVICE); ft = torch.tensor(Fa, dtype=torch.int32, device=DEVICE)
    fn = torch.cross(vt[ft[:, 1].long()] - vt[ft[:, 0].long()], vt[ft[:, 2].long()] - vt[ft[:, 0].long()], dim=1)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-12)
    l1 = torch.tensor([0.3, 0.8, 0.5], device=DEVICE); l1 /= l1.norm()
    l2 = torch.tensor([-0.6, 0.2, -0.8], device=DEVICE); l2 /= l2.norm()
    sh = 0.3 + 0.45 * (fn @ l1).abs() + 0.25 * (fn @ l2).abs()
    hom = torch.cat([vt, torch.ones(len(vt), 1, device=DEVICE)], 1)
    n = len(mvps); rows = (n + cols - 1) // cols
    canvas = Image.new("L", (cols * res, rows * res + 36), 255)
    for i in range(n):
        m = torch.as_tensor(mvps[i]).float().to(DEVICE)
        rast, _ = dr.rasterize(ctx, (hom @ m.T)[None].contiguous(), ft, (res, res))
        fid = rast[0, ..., 3].long(); img = torch.ones(res, res, device=DEVICE); msk = fid > 0
        img[msk] = sh[fid[msk] - 1]
        canvas.paste(Image.fromarray((img.cpu().numpy()[::-1] * 255).astype(np.uint8)), ((i % cols) * res, 36 + (i // cols) * res))
    d = ImageDraw.Draw(canvas)
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
    except Exception: font = ImageFont.load_default()
    d.text((10, 6), f"{TAG}  step {step:4d}/{STEPS}   V={len(Vn)} F={len(Fa)}   {n} training views", fill=0, font=font)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    canvas.save(f"{SNAPSHOT_DIR}/frame_{step:05d}.png")

def iou_fn(vv, ff):
    vt = torch.tensor(np.asarray(vv), dtype=torch.float32, device=DEVICE)
    ft = torch.tensor(np.asarray(ff, np.int32), dtype=torch.int32, device=DEVICE)
    return compute_iou_n(render_views_n(ctx, vt, ft, mvps), gt)

def report(tag, V, Fa):
    ho = heldout_exam(ctx, V, Fa); wt, _ = check_watertight(Fa)
    s = si_faces(V, Fa); f = fold_frac(V, Fa)
    print(f"[{tag}] V={len(V)} F={len(Fa)} watertight={wt} | train={iou_fn(V, Fa):.4f} "
          f"ho16={ho[0]:.4f} hair={ho[1]} | SI={s} ({100*s/len(Fa):.1f}%) folds={100*f:.1f}%",
          flush=True)
    return ho[0], s

# ---------------------------------------------------------------- optimize
ho0, si0 = report("base", V, Fa)
verts_t = torch.tensor(V, dtype=torch.float32, device=DEVICE).requires_grad_(True)
opt = torch.optim.Adam([verts_t], lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS, eta_min=LR_MIN)

def rebuild(Fa):
    faces_t = torch.tensor(Fa.astype(np.int32), dtype=torch.int32, device=DEVICE)
    src, dst, deg, excl = build_adj(Fa.astype(np.int32), int(Fa.max()) + 1, want_excl=False)
    pairs_t = torch.tensor(build_pairs(Fa.astype(np.int32)), device=DEVICE)
    return faces_t, faces_t.long(), src, dst, deg, pairs_t

faces_t, faces_l, src, dst, deg, pairs_t = rebuild(Fa)
t0 = time.time(); nflips_total = 0; ncollapse_total = 0; nsplit_total = 0; npush_total = 0
_E0 = np.concatenate([Fa[:, [0, 1]], Fa[:, [1, 2]], Fa[:, [2, 0]]]); me0 = float(np.linalg.norm(V[_E0[:, 0]] - V[_E0[:, 1]], axis=1).mean())
for step in range(STEPS):
    opt.zero_grad()
    me = mean_edge_of(verts_t.detach(), src, dst)
    sl = dl = fl = nl = torch.tensor(0.0, device=DEVICE)
    for i in range(NV):
        if MODE == "64v":
            sil, ndc_z, fg, diff = render_sdd(ctx, verts_t, faces_t, mvps[i], views[i])
            fl = fl + F.l1_loss(diff, gtdf_t[i])
            if W_NORMAL > 0:
                nl = nl + F.l1_loss(run_64v.render_normals(ctx, verts_t, faces_t, mvps[i], views[i]), gtn_t[i])
        else:
            sil, ndc_z, fg = render_sil_and_depth(ctx, verts_t, faces_t, mvps[i])
        sl = sl + F.l1_loss(sil[0], targets[i])
        dl = dl + depth_loss_masked(ndc_z, fg, gtd_t[i], gtfg_t[i])
    sl, dl, fl, nl = sl / NV, dl / NV, fl / NV, nl / NV
    loss = (sl + W_DEPTH * dl + W_DIFF * fl + W_NORMAL * nl
            + W_LAP * LAP_MULT * laplacian_loss(verts_t, faces_t)
            + W_EDGE * edge_length_loss(verts_t, faces_t)
            + W_QUAL * _qual_loss(verts_t, faces_t)
            + W_SPIKE * spike_pen(verts_t, src, dst, deg, me)
            + W_SLIVER * sliver_pen(verts_t, faces_l, me)
            + W_FOLD * FOLD_MULT * fold_loss(verts_t, faces_l, pairs_t)
            + W_T * field_loss(verts_t, faces_l))
    loss.backward()
    opt.step(); sched.step()
    if FLIP_EVERY > 0 and (step + 1) % FLIP_EVERY == 0 and step + 1 < STEPS:
        with torch.no_grad():
            Vn = verts_t.detach().cpu().numpy().astype(np.float64)
            nc = ns = 0
            if COLLAPSE_EVERY > 0 and (step + 1) % COLLAPSE_EVERY == 0:
                cap = int(COLLAPSE_FRAC * len(Fa)) if COLLAPSE_FRAC > 0 else COLLAPSE_MAX
                if ADAPT_REMESH:
                    # Curvature-adaptive remeshing (Palfinger-style target length, on DLFL ops).
                    # Per-vertex target L(v) = me0 * lerp(TMAX->TMIN, smoothstep(dihedral)):
                    # curved -> short target -> subdivide_edge; flat -> long target -> collapse.
                    # Fixes vertex migration (verts pile into flat/concave regions during the
                    # sphere->shape deformation and uniform subdivision locks that in).
                    def _target(Vx, Fx):
                        kap = vertex_dihedral(_taubin_np(Vx, Fx, ADAPT_SMOOTH_K) if ADAPT_SMOOTH_K > 0 else Vx, Fx)
                        t = np.clip((kap - ADAPT_LO) / (ADAPT_HI - ADAPT_LO), 0, 1); t = t * t * (3 - 2 * t)
                        _floor = ADAPT_LMIN_PX * PX_SIZE if "PX_SIZE" in globals() else 0.0
                        _Lmax = me0 * ADAPT_TMAX
                        _target.last_cthr = None
                        if ADAPT_MODE == "dunyach":
                            # Dunyach 2013 sizing field: L = sqrt(6 eps/k - 3 eps^2), eps in pixels -> tied to supervision
                            kmax = principal_kmax(_taubin_np(Vx, Fx, ADAPT_SMOOTH_K) if ADAPT_SMOOTH_K > 0 else Vx, Fx)
                            eps = ADAPT_EPS_PX * (PX_SIZE if "PX_SIZE" in globals() else me0 / 6)
                            Ld = np.sqrt(np.maximum(6 * eps / np.maximum(kmax, 1e-9) - 3 * eps * eps, 0.0))
                            Ld[kap > ADAPT_FOLD] = _Lmax
                            Ld = np.clip(Ld, _floor, _Lmax)
                            if ADAPT_GRADE > 0: Ld = grade_sizing(Vx, Fx, Ld, ADAPT_GRADE)
                            return Ld
                        if ADAPT_MODE == "velocity":
                            # Palfinger 2022: closed loop on vertex speed. Faster than median -> coarser, slower -> finer.
                            le = local_edge_len(Vx, Fx)
                            Vl = getattr(_target, "V_last", None); rl = getattr(_target, "ref_len", None)
                            if Vl is None or len(Vl) != len(Vx) or rl is None or len(rl) != len(Vx):
                                rl = le.copy(); nu = np.ones(len(Vx))
                            else:
                                nu = np.linalg.norm(Vx - Vl, axis=1) / max(me0, 1e-9)
                            nu_med = max(np.median(nu), 1e-9)
                            rl = rl * np.clip(1 + (nu / nu_med - 1) * ADAPT_NU_GAIN, 0.5, 2.0)
                            rl[kap > ADAPT_FOLD] = _Lmax
                            rl = np.clip(rl, _floor, _Lmax)
                            if ADAPT_GRADE > 0: rl = grade_sizing(Vx, Fx, rl, ADAPT_GRADE)
                            _target.ref_len = rl; _target.nu = nu
                            return rl
                        if ADAPT_MODE == "curv_uniform":
                            # 3DV-2026: split high-curvature edges; elsewhere uniform split by the average edge length
                            Lu = np.full(len(Vx), float(np.mean(local_edge_len(Vx, Fx))))
                            Lu[t > 0.5] = me0 * ADAPT_TMIN
                            Lu[kap > ADAPT_FOLD] = _Lmax
                            Lu = np.clip(Lu, _floor, _Lmax)
                            if ADAPT_GRADE > 0: Lu = grade_sizing(Vx, Fx, Lu, ADAPT_GRADE)
                            return Lu
                        if ADAPT_MODE == "residual":
                            ef, cf = face_residual(Vx, Fx)
                            rf = ef / np.maximum(cf, 1.0)                        # mean error per covered pixel-view (absolute, pixel scale)
                            ev = np.zeros(len(Vx))
                            for k in range(3): np.maximum.at(ev, Fx[:, k], rf)   # vertex = worst incident face
                            if ADAPT_ABS:
                                # policy: unfit -> refine; fitted+flat -> may coarsen; otherwise KEEP the current
                                # local edge length (the fit needs it). v7 quantiles labelled half the mesh
                                # "fitted" by construction and coarsened fertility to 2.4k V.
                                src_ = np.concatenate([Fx[:, 0], Fx[:, 1], Fx[:, 2], Fx[:, 1], Fx[:, 2], Fx[:, 0]])
                                dst_ = np.concatenate([Fx[:, 1], Fx[:, 2], Fx[:, 0], Fx[:, 0], Fx[:, 1], Fx[:, 2]])
                                le = np.zeros(len(Vx)); cnt_ = np.zeros(len(Vx))
                                np.add.at(le, src_, np.linalg.norm(Vx[src_] - Vx[dst_], axis=1)); np.add.at(cnt_, src_, 1)
                                le /= np.maximum(cnt_, 1)                                   # current local edge length
                                unfit = ev > ADAPT_R_SPLIT
                                coarse_ok = (ev < ADAPT_R_KEEP) & ((t < 0.05) if ADAPT_KEEP_FLAT else True)   # residual ~0 (and flat, if KEEP_FLAT)
                                Lk = le.copy()
                                Lk[unfit] = np.minimum(le[unfit] * 0.5, me0 * ADAPT_TMIN)   # halve where unfit
                                Lk[coarse_ok] = me0 * ADAPT_TMAX
                                Lk[kap > ADAPT_FOLD] = me0 * ADAPT_TMAX
                                _target.last_err = rf
                                Lk = np.maximum(Lk, ADAPT_LMIN_PX * PX_SIZE) if "PX_SIZE" in globals() else Lk
                                # separate collapse threshold: "keep" verts must not lose their shorter half
                                # (0.8 x local mean collapses half the incident edges) -> only slivers (< 0.4 x local)
                                cthr = 0.4 * le
                                cthr[coarse_ok | (kap > ADAPT_FOLD)] = ADAPT_CRATIO * me0 * ADAPT_TMAX
                                cthr[unfit] = ADAPT_CRATIO * Lk[unfit]
                                _target.last_cthr = cthr
                                if ADAPT_GRADE > 0: Lk = grade_sizing(Vx, Fx, Lk, ADAPT_GRADE)
                                return Lk
                            lo, hi = np.quantile(ev, ADAPT_E_LO), np.quantile(ev, ADAPT_E_HI)
                            te = np.clip((ev - lo) / max(hi - lo, 1e-9), 0, 1); te = te * te * (3 - 2 * te)
                            t = np.maximum(te, ADAPT_CURV_W * t)
                            _target.last_err = rf
                        t[kap > ADAPT_FOLD] = 0.0                    # tangles get the flat (long) target
                        Lt_ = me0 * (ADAPT_TMAX - (ADAPT_TMAX - ADAPT_TMIN) * t)
                        Lt_ = np.maximum(Lt_, _floor)
                        if ADAPT_GRADE > 0: Lt_ = grade_sizing(Vx, Fx, Lt_, ADAPT_GRADE)
                        return Lt_
                    Lt = _target(Vn, Fa)
                    if step + 1 == COLLAPSE_EVERY:
                        _px = globals().get("PX_SIZE", float("nan"))
                        _re = getattr(_target, "last_err", None)
                        _rs = f" | face resid/px median={np.median(_re):.3f} p90={np.percentile(_re,90):.3f} unfit(>{ADAPT_R_SPLIT})={100*(_re>ADAPT_R_SPLIT).mean():.0f}%" if _re is not None else ""
                        print(f"[adapt] cfg: me0={me0:.4f} PX_SIZE={_px:.4f} floor={ADAPT_LMIN_PX*_px:.4f} | L min/med/max={Lt.min():.4f}/{np.median(Lt):.4f}/{Lt.max():.4f} | mean edge now={np.linalg.norm(Vn[Fa[:,0]]-Vn[Fa[:,1]],axis=1).mean():.4f}{_rs}", flush=True)
                    tri = Vn[Fa]
                    el3 = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                                    np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                                    np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], 1)
                    ratio = el3.max(1) / Lt[Fa].min(1)          # longest edge vs target of most-curved corner
                    si_bad, si_frac = _si_ring_mask(Vn, Fa)
                    ratio[si_bad] = 0.0                          # never refine a tangle or its ring
                    if ADAPT_VEL_GUARD > 0:
                        _Vl = getattr(_target, "V_last", None)
                        if _Vl is not None and len(_Vl) == len(Vn):
                            _nu = np.linalg.norm(Vn - _Vl, axis=1); _mov = _nu > ADAPT_VEL_GUARD * max(np.median(_nu), 1e-12)
                            _fm = _mov[Fa].any(1); ratio[_fm] = 0.0
                            print(f"[adapt] step {step+1}: velocity guard excluded {100*_fm.mean():.0f}% of faces (still moving)", flush=True)
                    gate_open = si_frac <= ADAPT_SI_GATE and len(Fa) < ADAPT_MAX_F
                    if not gate_open:
                        print(f"[adapt] step {step+1}: splits skipped (SI {100*si_frac:.1f}% > gate {100*ADAPT_SI_GATE:.0f}% or F>={ADAPT_MAX_F}); collapse/flip only", flush=True)
                    fids = np.where(ratio > ADAPT_SRATIO)[0] if gate_open else np.zeros(0, int)
                    if len(fids):
                        fids = fids[np.argsort(-ratio[fids])][:max(1, int(ADAPT_SPLIT_FRAC * len(Fa)))].tolist()
                        from phase1c_pipeline import dlfl_subdivide_arrays
                        cy = Vn[Fa[fids]].mean(1)[:, 1]
                        hist, _ = np.histogram(cy, bins=np.linspace(Vn[:, 1].min(), Vn[:, 1].max(), 7))
                        print(f"[adapt] step {step+1}: split {len(fids)} faces, y-bands bottom->top {hist.tolist()}", flush=True)
                        Vn, Fa, ns = dlfl_subdivide_arrays(Vn, Fa, fids, expand_ring=False)
                        wt_, _ = check_watertight(Fa); assert wt_
                        Lt = _target(Vn, Fa)
                    # Botsch-Kobbelt converges only if split and collapse both run to completion each pass;
                    # a 2 % cap on both is asymmetric (a split adds ~4 verts, a collapse removes 1) -> linear growth
                    # to the face cap regardless of L. Tangle safety comes from the SI-ring exclusion + gate, not the cap.
                    _t0 = time.time()
                    _cthr = getattr(_target, "last_cthr", None)
                    Vn, Fa, nc = collapse_short_edges(Vn, Fa, COLLAPSE_RATIO, int(ADAPT_COLLAPSE_FRAC * len(Fa)), vthr=(_cthr if (_cthr is not None and len(_cthr) == len(Vn)) else ADAPT_CRATIO * Lt))
                    print(f"[adapt] step {step+1}: +{ns} split edges, -{nc} collapses -> V={len(Vn)} F={len(Fa)} ({time.time()-_t0:.0f}s collapse)", flush=True)
                    _target.V_last = Vn.copy()
                    if getattr(_target, 'ref_len', None) is not None and len(_target.ref_len) != len(Vn): _target.ref_len = None
                    nsplit_total += ns
                else:
                    Vn, Fa, nc = collapse_short_edges(Vn, Fa, COLLAPSE_RATIO, cap, thr_abs=(COLLAPSE_RATIO * me0) if COLLAPSE_ABS else None)
                ncollapse_total += nc
            Vn, Fa, nf = flip_sweep(Vn, Fa, passes=3)
            if SMOOTH_ITERS > 0:
                Vn = tangential_smooth(Vn, Fa, SMOOTH_ITERS, SMOOTH_LAM)
            if SI_PUSH > 0 and (step + 1) % SI_EVERY == 0:
                # residual overlaps are non-adjacent near-parallel faces: nudge
                # each intersecting pair apart along the mean normal (delta =
                # SI_PUSH x mean edge); DR/target losses pull the shape back.
                om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(Vn),
                                               o3d.utility.Vector3iVector(Fa.astype(np.int32)))
                prs = np.asarray(om.get_self_intersecting_triangles())
                if len(prs):
                    tri = Vn[Fa]; cen = tri.mean(1)
                    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
                    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
                    me_np = np.linalg.norm(Vn[Fa[:, 0]] - Vn[Fa[:, 1]], axis=1).mean()
                    disp = np.zeros_like(Vn); cnt = np.zeros(len(Vn))
                    for a, b in prs:
                        n = nrm[a] if abs(nrm[a] @ nrm[b]) > 0.5 else nrm[a] + nrm[b]
                        n /= np.linalg.norm(n) + 1e-12
                        s = np.sign((cen[b] - cen[a]) @ n) or 1.0
                        disp[Fa[a]] -= s * n * SI_PUSH * me_np; cnt[Fa[a]] += 1
                        disp[Fa[b]] += s * n * SI_PUSH * me_np; cnt[Fa[b]] += 1
                    m = cnt > 0
                    Vn[m] += disp[m] / cnt[m, None]
                    npush_total += int(len(prs))
            nflips_total += nf
            if nc > 0 or ns > 0:
                # vertex count changed: new parameter tensor + fresh Adam at current lr
                cur_lr = opt.param_groups[0]["lr"]
                V = Vn
                verts_t = torch.tensor(Vn, dtype=torch.float32, device=DEVICE).requires_grad_(True)
                opt = torch.optim.Adam([verts_t], lr=cur_lr)
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                    opt, T_max=max(1, STEPS - step - 1), eta_min=LR_MIN)
                faces_t, faces_l, src, dst, deg, pairs_t = rebuild(Fa)
            else:
                if SMOOTH_ITERS > 0 or SI_PUSH > 0:
                    verts_t.data.copy_(torch.tensor(Vn, dtype=torch.float32, device=DEVICE))
                if nf > 0:
                    faces_t, faces_l, src, dst, deg, pairs_t = rebuild(Fa)
    if SNAPSHOT_EVERY > 0 and step % SNAPSHOT_EVERY == 0:
        _snapshot(step, verts_t.detach().cpu().numpy().astype(np.float64), Fa)
    if (step + 1) % 100 == 0:
        Vn = verts_t.detach().cpu().numpy().astype(np.float64)
        s = si_faces(Vn, Fa)
        print(f"[step {step+1}/{STEPS}] sil={sl.item():.4f} flips={nflips_total} collapses={ncollapse_total} splits={nsplit_total} pushes={npush_total} V={len(Fa) and len(Vn)} "
              f"SI={s} ({100*s/len(Fa):.1f}%) folds={100*fold_frac(Vn, Fa):.1f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

V = verts_t.detach().cpu().numpy().astype(np.float64)
V, Fa, nf = flip_sweep(V, Fa, passes=4); nflips_total += nf
wt, nbad = check_watertight(Fa); assert wt, nbad
hof, sif = report("final", V, Fa)
print(f"[p4] ho16 {ho0:.4f} -> {hof:.4f} ({(hof-ho0)*100:+.2f}) | SI {100*si0/len(Fa):.1f}% -> "
      f"{100*sif/len(Fa):.1f}% | total flips={nflips_total}", flush=True)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
print(f"[p4] saved cow_{SHAPE}_{TAG}.npz", flush=True)
