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
FOLD_MULT = float(os.environ.get("FOLD_MULT", "1.0"))
COLLAPSE_EVERY = int(os.environ.get("COLLAPSE_EVERY", "0"))   # 0 = off
COLLAPSE_RATIO = float(os.environ.get("COLLAPSE_RATIO", "0.3"))
ADAPT_REMESH = int(os.environ.get("ADAPT_REMESH", "0"))   # curvature-adaptive remeshing on DLFL ops (split where curved, collapse where flat)
ADAPT_LO = float(os.environ.get("ADAPT_LO", "8.0"))        # dihedral (deg) at/below which a vertex counts as flat
ADAPT_HI = float(os.environ.get("ADAPT_HI", "30.0"))       # dihedral at/above which a vertex counts as fully curved
ADAPT_TMIN = float(os.environ.get("ADAPT_TMIN", "0.5"))    # target edge length (x me0) in curved regions
ADAPT_TMAX = float(os.environ.get("ADAPT_TMAX", "1.6"))    # target edge length (x me0) in flat regions
ADAPT_CRATIO = float(os.environ.get("ADAPT_CRATIO", "0.5"))# collapse edges shorter than CRATIO x local target
ADAPT_SPLIT_FRAC = float(os.environ.get("ADAPT_SPLIT_FRAC", "0.02"))  # cap: faces split per pass as a fraction of F
ADAPT_MAX_F = int(os.environ.get("ADAPT_MAX_F", "60000"))   # stop splitting above this face count (256^2 supervision ceiling; pure-Python DLFL cost)
ADAPT_FOLD = float(os.environ.get("ADAPT_FOLD", "70.0"))    # dihedral above this = tangle/fold, not a feature: never split, let collapse clean it
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
    gt, gtd, gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
    cow_v13.N_VIEWS = 64
    HF = build_vote_hull(ctx, mvps, gvn, gf_gt, V, DEVICE, nres=256, hires=512, vote=2)
    DEAD = 1.0 * HF.pitch
    gtdf_t = [torch.from_numpy(gtdiff[i]).float().to(DEVICE) for i in range(64)]
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
    sl = dl = fl = torch.tensor(0.0, device=DEVICE)
    for i in range(NV):
        if MODE == "64v":
            sil, ndc_z, fg, diff = render_sdd(ctx, verts_t, faces_t, mvps[i], views[i])
            fl = fl + F.l1_loss(diff, gtdf_t[i])
        else:
            sil, ndc_z, fg = render_sil_and_depth(ctx, verts_t, faces_t, mvps[i])
        sl = sl + F.l1_loss(sil[0], targets[i])
        dl = dl + depth_loss_masked(ndc_z, fg, gtd_t[i], gtfg_t[i])
    sl, dl, fl = sl / NV, dl / NV, fl / NV
    loss = (sl + W_DEPTH * dl + W_DIFF * fl
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
                        kap = vertex_dihedral(Vx, Fx)
                        t = np.clip((kap - ADAPT_LO) / (ADAPT_HI - ADAPT_LO), 0, 1); t = t * t * (3 - 2 * t)
                        t[kap > ADAPT_FOLD] = 0.0                    # tangles get the flat (long) target
                        return me0 * (ADAPT_TMAX - (ADAPT_TMAX - ADAPT_TMIN) * t)
                    Lt = _target(Vn, Fa)
                    tri = Vn[Fa]
                    el3 = np.stack([np.linalg.norm(tri[:, 1] - tri[:, 0], axis=1),
                                    np.linalg.norm(tri[:, 2] - tri[:, 1], axis=1),
                                    np.linalg.norm(tri[:, 0] - tri[:, 2], axis=1)], 1)
                    ratio = el3.max(1) / Lt[Fa].min(1)          # longest edge vs target of most-curved corner
                    fids = np.where(ratio > 1.0)[0] if len(Fa) < ADAPT_MAX_F else np.zeros(0, int)
                    if len(fids):
                        fids = fids[np.argsort(-ratio[fids])][:max(1, int(ADAPT_SPLIT_FRAC * len(Fa)))].tolist()
                        from phase1c_pipeline import dlfl_subdivide_arrays
                        Vn, Fa, ns = dlfl_subdivide_arrays(Vn, Fa, fids, expand_ring=False)
                        wt_, _ = check_watertight(Fa); assert wt_
                        Lt = _target(Vn, Fa)
                    Vn, Fa, nc = collapse_short_edges(Vn, Fa, COLLAPSE_RATIO, cap, vthr=ADAPT_CRATIO * Lt)
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
