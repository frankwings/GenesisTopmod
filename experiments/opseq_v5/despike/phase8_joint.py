"""phase8_joint.py — Joint optimisation: mesh geometry + appearance + camera poses.

v1: per-vertex colour, silhouette + unshaded photometric
v2: + mono-depth prior, shading, normal-consistency, anti-crumpling
v3: + UV texture (coarse-to-fine mip bias), camera recovery test, pose-consistency

Usage (v3 gate — camera recovery with texture, geometry frozen):
  REAL_DATA=real/dino3 SHAPE=dino TAGP=recov_tex1 TEX=1 JOINT_CAM=1 FREEZE_GEO=1 \\
    CAM_PERTURB_DEG=0.3 STEPS_B=2000 W_CAM=0.1 LR_CAM=5e-4 \\
    python3 despike/phase8_joint.py dino_joint_dgrad

Usage (v3 three-way joint):
  REAL_DATA=real/dino3 SHAPE=dino TAGP=v3_cam1 TEX=1 JOINT_CAM=1 STEPS_B=2500 W_CAM=0.1 \\
    W_DEPTH_MONO=1 W_DEPTH_L1=0.2 W_DEPTH_GRAD=1.0 W_SIL=6 SHADE=0 W_PHO=0.3 LR_V=1e-4 \\
    python3 despike/phase8_joint.py dino_joint_dgrad
"""
import sys, os, math, time, json, subprocess
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from cow_v13 import (build_adj, build_pairs, fold_loss, mean_edge_of,
                     spike_pen, sliver_pen, DEVICE, W_SPIKE, W_SLIVER, W_FOLD, W_LAP as _W_LAP)
from eval_local_refine import laplacian_loss, edge_length_loss, W_LAP, W_EDGE
from phase1b_pipeline import check_watertight

# ── Environment ──────────────────────────────────────────────────────────────
REAL_DATA     = os.environ.get("REAL_DATA", "real/dino3")
SHAPE         = os.environ.get("SHAPE", "dino")
TAGP          = os.environ.get("TAGP", "joint")
TRAIN_RES     = int(os.environ.get("TRAIN_RES", "512"))
JOINT_CAM     = int(os.environ.get("JOINT_CAM", "0"))   # 0=freeze (v2 default)
FLIP_EVERY    = int(os.environ.get("FLIP_EVERY", "0"))
SIL_BLUR      = float(os.environ.get("SIL_BLUR", "0"))
HULL_VOTE     = int(os.environ.get("HULL_VOTE", "8"))
HULL_DEAD     = float(os.environ.get("HULL_DEAD", "3.0"))
STEPS_A       = int(os.environ.get("STEPS_A", "300"))
STEPS_B       = int(os.environ.get("STEPS_B", "1500"))
# ── v2 hypers ──
W_DEPTH_MONO  = float(os.environ.get("W_DEPTH_MONO", "1.0"))
W_DEPTH_L1    = float(os.environ.get("W_DEPTH_L1", "1.0"))    # absolute (scale-shift aligned) term: global shape, biased by the mono model
W_DEPTH_GRAD  = float(os.environ.get("W_DEPTH_GRAD", "0.5"))  # gradient-matching term: local relief (arms, mouth, base step)
SHADE         = int(os.environ.get("SHADE", "1"))         # 1=shaded photometric
AMB           = float(os.environ.get("AMB", "0.5"))
W_NC          = float(os.environ.get("W_NC", "0.05"))
# ── loss weights ──
W_PHO         = float(os.environ.get("W_PHO", "0.3"))     # v2 default (was 1.0)
W_CLAP        = float(os.environ.get("W_CLAP", "0.05"))
W_SIL         = float(os.environ.get("W_SIL", "3.0"))
W_CAM         = float(os.environ.get("W_CAM", "1.0"))
W_T           = float(os.environ.get("W_T", "2.0"))
PHO_CLIP      = float(os.environ.get("PHO_CLIP", "0.3"))
SIG_R_DEG     = float(os.environ.get("SIG_R_DEG", "1.0"))
SIG_T         = float(os.environ.get("SIG_T", "0.01"))
LR_V          = float(os.environ.get("LR_V", "1e-4"))     # v2 default (was 3e-4)
LR_C          = float(os.environ.get("LR_C", "1e-2"))
LR_CAM        = float(os.environ.get("LR_CAM", "2e-4"))
W_LAP_J       = float(os.environ.get("W_LAP_J", str(5.0 * W_LAP)))   # v2: 5x anti-crumple
W_EDGE_J      = float(os.environ.get("W_EDGE_J", str(W_EDGE)))
# ── v3 hypers ──
TEX           = int(os.environ.get("TEX", "0"))           # 1=UV texture, 0=vertex colour
TEX_RES       = int(os.environ.get("TEX_RES", "2048"))
W_TV          = float(os.environ.get("W_TV", "1e-3"))     # texture TV regulariser
LR_T          = float(os.environ.get("LR_T", "1e-2"))     # texture LR
CAM_PERTURB_DEG = float(os.environ.get("CAM_PERTURB_DEG", "0"))  # camera perturbation (recovery test)
CAM_PERTURB_SEED = int(os.environ.get("CAM_PERTURB_SEED", "0"))
FREEZE_GEO    = int(os.environ.get("FREEZE_GEO", "0"))    # 1=freeze vertex positions
LR_CAM_INIT   = float(os.environ.get("LR_CAM_INIT", "0")) # >0: initial camera LR, step down to LR_CAM at 60%

os.makedirs("despike/results_genus", exist_ok=True)

mesh_stem = sys.argv[1] if len(sys.argv) > 1 else f"{SHAPE}_real7_auto"
TAG = f"{SHAPE}_joint_{TAGP}"
print(f"[p8] SHAPE={SHAPE} TAGP={TAGP} JOINT_CAM={JOINT_CAM} TEX={TEX} FREEZE_GEO={FREEZE_GEO} "
      f"W_DEPTH_MONO={W_DEPTH_MONO} SHADE={SHADE} CAM_PERTURB_DEG={CAM_PERTURB_DEG} "
      f"mesh={mesh_stem}", flush=True)

# ── Load mesh ─────────────────────────────────────────────────────────────────
mesh_path = f"despike/results_genus/{mesh_stem}.npz"
z = np.load(mesh_path)
V0 = z["verts"].astype(np.float32)
F0 = z["tris"].astype(np.int32)
print(f"[p8] input mesh: V={len(V0)} F={len(F0)}", flush=True)

# ── UV atlas (v3, TEX=1) ─────────────────────────────────────────────────────
if TEX:
    import xatlas
    uv_cache = mesh_path.replace('.npz', '_uv.npz')
    if os.path.exists(uv_cache):
        _d = np.load(uv_cache)
        _vmapping = _d['vmapping']
        _uv_indices = _d['uv_idx'] if 'uv_idx' in _d else _d['indices']
        _uvs = _d['uvs']
        print(f"[p8] loaded UV cache: {uv_cache} ({len(_uvs)} atlas verts)", flush=True)
    else:
        print(f"[p8] computing xatlas UV parametrization ...", flush=True)
        _t_uv = time.time()
        _vmapping, _uv_indices, _uvs = xatlas.parametrize(V0, F0)
        np.savez(uv_cache, vmapping=_vmapping, indices=_uv_indices, uvs=_uvs)
        print(f"[p8] xatlas UV done: {len(_uvs)} atlas verts, {time.time()-_t_uv:.1f}s -> {uv_cache}", flush=True)
    uv_idx_t = torch.tensor(_uv_indices.astype(np.int32), dtype=torch.int32, device=DEVICE)
    uvs_t    = torch.tensor(_uvs.astype(np.float32), dtype=torch.float32, device=DEVICE)  # [N_atlas, 2]
    # Learnable texture  [1, TEX_RES, TEX_RES, 3]
    T_tex = torch.full([1, TEX_RES, TEX_RES, 3], 0.5, device=DEVICE, dtype=torch.float32,
                       requires_grad=True)
    print(f"[p8] texture: [{TEX_RES}x{TEX_RES}] init 0.5, atlas verts={len(_uvs)}", flush=True)
else:
    uv_idx_t = uvs_t = T_tex = None

# ── Load scene ────────────────────────────────────────────────────────────────
os.environ["TRAIN_RES"] = str(TRAIN_RES)
from real_scene import load_real_scene
ctx = dr.RasterizeCudaContext()
scene = load_real_scene(REAL_DATA, DEVICE, res=TRAIN_RES)

with torch.no_grad():
    diff = (torch.bmm(scene.proj, scene.view_w2c) - scene.mvps).abs().max().item()
    print(f"[p8] MVP sanity: max|proj@view_w2c - mvps| = {diff:.2e}", flush=True)
    assert diff < 1e-4, f"MVP decomp error {diff}"

N_TRAIN = len(scene.mvps)
res = scene.res
HAS_DEPTH = scene._has_depth and W_DEPTH_MONO > 0
print(f"[p8] depth prior: {'ON' if HAS_DEPTH else 'OFF'} | shading: {'ON' if SHADE else 'OFF'}", flush=True)

# ── Hull field ────────────────────────────────────────────────────────────────
HF = scene.hull(ctx, extra_pts=V0, nres=256, hires=512, vote=HULL_VOTE)
DEAD = HULL_DEAD * HF.pitch
def field_dist(pts): return F.relu(HF.dist(pts) - DEAD)

# ── Stacked GT / photo / mono ──────────────────────────────────────────────────
import batch_losses as _bl
_bl.VALID = torch.from_numpy(scene.valid.astype(np.float32)).to(DEVICE)
gt_np    = scene.gt                                            # [N,H,W] uint8, 0=fg
gt_t     = torch.from_numpy((gt_np < 128).astype(np.float32)).to(DEVICE)
from batch_losses import soft_targets as _soft
gt_soft  = _soft(gt_t, SIL_BLUR)
valid_t  = torch.from_numpy(scene.valid.astype(np.float32)).to(DEVICE)
photo_t  = torch.from_numpy(scene.rgb).to(DEVICE)             # [N,H,W,3]
ho_photo = torch.from_numpy(scene.ho_rgb).to(DEVICE)
mono_t   = torch.from_numpy(scene.mono).to(DEVICE)            # [N,H,W] float32
ho_mono  = torch.from_numpy(scene.ho_mono).to(DEVICE)

def erode_mask_batch(mask_b, n_px=3):
    k  = 2 * n_px + 1
    m  = mask_b.float().unsqueeze(1)
    m  = -F.max_pool2d(-m, kernel_size=k, stride=1, padding=n_px)
    return m[:, 0] > 0.5

gt_interior  = erode_mask_batch(gt_t > 0.5)
ho_gt_t      = (torch.from_numpy(scene.ho_gt).to(DEVICE) < 128)
ho_interior  = erode_mask_batch(ho_gt_t)


# ═══════════════════════════════════════════════════════════════════════════════
# Geometry helpers
# ═══════════════════════════════════════════════════════════════════════════════

def compute_vertex_normals_grad(verts_t, faces_l):
    """Area-weighted vertex normals; gradient flows to verts_t."""
    v0 = verts_t[faces_l[:, 0]]
    v1 = verts_t[faces_l[:, 1]]
    v2 = verts_t[faces_l[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)       # [F,3] area-weighted
    nv = verts_t.shape[0]
    vn = verts_t.new_zeros(nv, 3)
    vn = vn.index_add(0, faces_l[:, 0], fn)
    vn = vn.index_add(0, faces_l[:, 1], fn)
    vn = vn.index_add(0, faces_l[:, 2], fn)
    norm = vn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return vn / norm                                   # [V,3]


def normal_consistency_loss(verts_t, faces_l, pairs_t):
    """Mean(1 - n_i·n_j) over adjacent face pairs — penalises crumpling."""
    v0 = verts_t[faces_l[:, 0]]
    v1 = verts_t[faces_l[:, 1]]
    v2 = verts_t[faces_l[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = fn / fn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ni = fn[pairs_t[:, 0]]; nj = fn[pairs_t[:, 1]]
    return (1.0 - (ni * nj).sum(-1)).clamp(min=0).mean()


@torch.no_grad()
def crumpling_metric(verts_np, tris_np):
    """Mean adjacent-face angle (degrees) — lower = smoother."""
    from cow_v13 import build_pairs as _bp
    pairs = _bp(tris_np)
    if len(pairs) == 0:
        return 0.0
    vt = torch.tensor(verts_np, device=DEVICE)
    ft = torch.tensor(tris_np, dtype=torch.long, device=DEVICE)
    pt = torch.tensor(pairs, device=DEVICE)
    v0 = vt[ft[:, 0]]; v1 = vt[ft[:, 1]]; v2 = vt[ft[:, 2]]
    fn = torch.cross(v1 - v0, v2 - v0, dim=-1)
    fn = fn / fn.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    ni = fn[pt[:, 0]]; nj = fn[pt[:, 1]]
    cos_ab = (ni * nj).sum(-1).clamp(-1 + 1e-7, 1 - 1e-7)
    angles = torch.acos(cos_ab) * 180.0 / math.pi
    return float(angles.mean().item())


# ═══════════════════════════════════════════════════════════════════════════════
# SE(3) camera corrections (v1 code; cameras frozen in v2 by default)
# ═══════════════════════════════════════════════════════════════════════════════

def so3_exp_batch(omega):
    theta = omega.norm(dim=-1).clamp(min=1e-8)
    k = omega / theta.unsqueeze(-1)
    Z = torch.zeros_like(k[:, 0])
    K = torch.stack([
        torch.stack([ Z,      -k[:, 2],  k[:, 1]], -1),
        torch.stack([ k[:, 2],  Z,      -k[:, 0]], -1),
        torch.stack([-k[:, 1],  k[:, 0],  Z      ], -1),
    ], -2)
    I3 = torch.eye(3, device=omega.device).unsqueeze(0)
    s  = theta.view(-1, 1, 1)
    return I3 + torch.sin(s) * K + (1 - torch.cos(s)) * torch.bmm(K, K)


def build_delta(omega_t, t_cam_t):
    N = omega_t.shape[0]
    R = so3_exp_batch(omega_t)
    delta = torch.eye(4, device=omega_t.device).unsqueeze(0).expand(N, -1, -1).clone()
    delta[:, :3, :3] = R
    delta[:, :3, 3]  = t_cam_t
    return delta


def eff_mvps(omega_t, t_cam_t):
    delta = build_delta(omega_t, t_cam_t)
    return torch.bmm(torch.bmm(scene.proj, delta), scene.view_w2c)


# ═══════════════════════════════════════════════════════════════════════════════
# v3 helpers: mip bias schedule, texture TV loss, pose consistency
# ═══════════════════════════════════════════════════════════════════════════════

def mip_bias_schedule(step, total_steps, B0=5.0):
    """Mip bias: B0 -> 0 linearly over first 60% of steps, then 0."""
    frac = min(step / max(0.6 * total_steps, 1), 1.0)
    return B0 * (1.0 - frac)


def texture_tv_loss(T):
    """Total variation on [1,H,W,3] texture."""
    dx = T[:, :, 1:, :] - T[:, :, :-1, :]
    dy = T[:, 1:, :, :] - T[:, :-1, :, :]
    return dx.pow(2).mean() + dy.pow(2).mean()


@torch.no_grad()
def pose_consistency_metric(mvps_eval, gt_masks_np, lo, hi, nres=256, dilate_px=15):
    """Carve hull from GT masks with mvps_eval, project back, report per-view coverage.
    Returns median coverage (higher = better pose consistency)."""
    from hull_field import carve_hull
    N = mvps_eval.shape[0]
    fgs = [torch.from_numpy((gt_masks_np[i] < 128).astype(bool)).to(DEVICE) for i in range(N)]
    hull = carve_hull(fgs, mvps_eval, lo, hi, nres, res, 2, DEVICE)
    # Hull voxel centres
    axes = [np.linspace(float(lo[a]), float(hi[a]), nres) for a in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing='ij')
    pts = np.stack([gx, gy, gz], -1)[hull]  # [K,3]
    if len(pts) == 0:
        return 0.0, [0.0] * N
    P = torch.tensor(pts, dtype=torch.float32, device=DEVICE)
    Ph = torch.cat([P, torch.ones(len(P), 1, device=DEVICE)], 1)  # [K,4]
    covs = []
    for i in range(N):
        clip = (mvps_eval[i] @ Ph.T).T
        w = clip[:, 3].clamp(min=1e-8)
        x_ndc, y_ndc = clip[:, 0] / w, clip[:, 1] / w
        ui = ((x_ndc + 1) * 0.5 * res).long().clamp(0, res - 1)
        vi = ((y_ndc + 1) * 0.5 * res).long().clamp(0, res - 1)
        inb = (x_ndc.abs() <= 1) & (y_ndc.abs() <= 1)
        hull_img = torch.zeros(res, res, dtype=torch.float32, device=DEVICE)
        hull_img[vi[inb], ui[inb]] = 1.0
        # Dilate (max_pool, kernel = 2*dilate_px+1)
        k = 2 * dilate_px + 1
        hull_d = F.max_pool2d(hull_img[None, None], kernel_size=k, stride=1, padding=dilate_px)[0, 0]
        mask_i = torch.from_numpy((gt_masks_np[i] < 128).astype(np.float32)).to(DEVICE)
        covered = (hull_d * mask_i).sum()
        total = mask_i.sum().clamp(min=1)
        covs.append(float(covered / total))
    return float(np.median(covs)), covs


# ═══════════════════════════════════════════════════════════════════════════════
# Batched renderer — silhouette + colour/texture (+shade) + inverse depth
# ═══════════════════════════════════════════════════════════════════════════════

def render_batch(verts_t, faces_t, faces_l, mvps_eff,
                 C_t=None, T_t=None, uvs_t_=None, uv_idx_t_=None, mip_bias_val=0.0,
                 view_w2c_eff=None, do_shade=True, do_depth=True):
    """Return sil [N,H,W], col_out [N,H,W,3], col_unshaded [N,H,W,3],
    rast_fg [N,H,W], inv_depth [N,H,W] or None.
    TEX mode: pass T_t, uvs_t_, uv_idx_t_, mip_bias_val.
    Vertex-colour mode: pass C_t."""
    N_v = verts_t.shape[0]
    N   = mvps_eff.shape[0]
    ones_V = torch.ones(N_v, 1, dtype=verts_t.dtype, device=verts_t.device)
    vh = torch.cat([verts_t, ones_V], -1)                    # [V,4]
    pos = (mvps_eff @ vh.T).transpose(1, 2).contiguous()     # [N,V,4]
    rast, rast_db = dr.rasterize(ctx, pos, faces_t, resolution=[res, res])

    # Silhouette
    ones_attr = torch.ones(N, N_v, 3, dtype=torch.float32, device=verts_t.device)
    sil_col, _ = dr.interpolate(ones_attr, rast, faces_t)
    sil = dr.antialias(sil_col.contiguous(), rast, pos, faces_t)[..., 0]
    rast_fg = rast[..., 3] > 0

    # Colour / texture
    if T_t is not None and uvs_t_ is not None:
        # Texture mode (v3)
        uv_exp = uvs_t_.unsqueeze(0).expand(N, -1, -1).contiguous()  # [N, N_atlas, 2]
        uv_img, uv_da = dr.interpolate(uv_exp, rast, uv_idx_t_,
                                        rast_db=rast_db, diff_attrs='all')
        mip_bias_t = torch.full([N, res, res], mip_bias_val,
                               device=verts_t.device) if mip_bias_val != 0.0 else None
        col_img = dr.texture(T_t, uv_img, uv_da,
                             filter_mode='linear-mipmap-linear',
                             mip_level_bias=mip_bias_t)               # [N,H,W,3]
        col_img = dr.antialias(col_img.contiguous(), rast, pos, faces_t)
    else:
        # Vertex colour mode (v1/v2)
        C_exp = C_t.unsqueeze(0).expand(N, -1, -1).contiguous()  # [N,V,3]
        col_img, _ = dr.interpolate(C_exp, rast, faces_t)
        col_img = dr.antialias(col_img.contiguous(), rast, pos, faces_t)

    # Shading (SHADE=1 and do_shade)
    if do_shade and SHADE and view_w2c_eff is not None:
        vn     = compute_vertex_normals_grad(verts_t, faces_l)       # [V,3]
        R      = view_w2c_eff[:, :3, :3]                             # [N,3,3]
        n_cam  = torch.bmm(R, vn.unsqueeze(0).expand(N,-1,-1).transpose(1,2)).transpose(1,2)
        n_cam  = (n_cam / n_cam.norm(dim=-1, keepdim=True).clamp(min=1e-8)).contiguous()
        n_img, _ = dr.interpolate(n_cam, rast, faces_t)              # [N,H,W,3]
        n_img  = n_img / n_img.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        shade  = AMB + (1.0 - AMB) * n_img[..., 2].clamp(min=0)     # headlight dot +Z
        col_out = (col_img * shade.unsqueeze(-1)).clamp(0, 1)
    else:
        col_out = col_img

    # Inverse depth  q = 1/z_cam,  z_cam = -(V_world → cam)[z] > 0
    if do_depth and view_w2c_eff is not None:
        z_cam      = -(view_w2c_eff @ vh.T)[:, 2, :]                # [N,V]
        inv_z_v    = (1.0 / z_cam.clamp(min=1e-6)).unsqueeze(-1)    # [N,V,1]
        inv_d_img, _ = dr.interpolate(inv_z_v.contiguous(), rast, faces_t)  # [N,H,W,1]
        inv_depth    = inv_d_img[..., 0]                             # [N,H,W]
    else:
        inv_depth = None

    return sil, col_out, col_img, rast_fg, inv_depth


# ═══════════════════════════════════════════════════════════════════════════════
# Loss functions
# ═══════════════════════════════════════════════════════════════════════════════

def colour_lap_loss(C_t, src, dst, deg):
    """deg shape [nv,1] from build_adj — do NOT unsqueeze again."""
    mean_nb = torch.zeros_like(C_t)
    mean_nb.index_add_(0, src, C_t[dst])
    mean_nb = mean_nb / deg.clamp(min=1)
    return (C_t - mean_nb).pow(2).mean()


def pho_loss_fn(col_out, rast_fg, interior, valid_t):
    """Robust photometric L1 on interior rendered+mask pixels."""
    mask   = rast_fg & interior
    if valid_t is not None:
        mask = mask & (valid_t > 0.5)
    mask_f = mask.float()
    denom  = mask_f.sum().clamp(min=1.0)
    err    = (col_out - photo_t).abs().mean(-1).clamp(max=PHO_CLIP)
    return (err * mask_f).sum() / denom


def depth_mono_loss_fn(inv_depth, rast_fg, interior, valid_t, mono_batch):
    """Scale-shift invariant depth loss with MiDaS gradient matching at 2 scales.
    inv_depth: [N,H,W] rendered inverse depth (carries gradient)
    mono_batch: [N,H,W] cached DA2 maps (no gradient)
    Returns (scalar_loss, n_skipped).
    """
    N = inv_depth.shape[0]
    total = torch.zeros(1, device=DEVICE)[0]
    n_skipped = 0

    for i in range(N):
        mask = rast_fg[i] & interior[i]
        if valid_t is not None:
            mask = mask & (valid_t[i] > 0.5)
        n_px = int(mask.float().sum().item())
        if n_px < 500:
            n_skipped += 1
            continue

        q   = inv_depth[i]        # [H,W], has gradient
        m   = mono_batch[i]       # [H,W], no gradient

        # Closed-form scale-shift alignment: detached solve
        with torch.no_grad():
            q_d  = q.detach()
            q_m  = q_d[mask]; m_m = m[mask]
            m_c  = m_m - m_m.mean()
            q_c  = q_m - q_m.mean()
            denom_ss = (m_c * m_c).sum().clamp(min=1e-8)
            s    = (m_c * q_c).sum() / denom_ss
            b    = q_m.mean() - s * m_m.mean()

        if float(s.item()) <= 0:
            n_skipped += 1
            continue

        # Residual map (has gradient through q)
        res_full = (s * m + b - q) * mask.float()   # [H,W]; 0 outside mask

        # L1 normalised loss
        denom_q  = q.detach()[mask].abs().mean().clamp(min=1e-8)
        l_l1     = res_full[mask].abs().mean() / denom_q

        # Gradient-matching at 2 scales (weight 0.5 each → total 0.5)
        def _grad_loss(rm, mk):
            gx   = (rm[:, 1:] - rm[:, :-1]).abs()
            gy   = (rm[1:, :] - rm[:-1, :]).abs()
            mkx  = mk[:, 1:] & mk[:, :-1]
            mky  = mk[1:, :] & mk[:-1, :]
            c    = (mkx.float().sum() + mky.float().sum()).clamp(min=1)
            return ((gx * mkx.float()).sum() + (gy * mky.float()).sum()) / c / denom_q

        gl1 = _grad_loss(res_full, mask)
        # Scale 2: avg-pool down by 2
        res_d  = F.avg_pool2d(res_full.unsqueeze(0).unsqueeze(0), 2)[0, 0]
        mask_d = (-F.max_pool2d(-mask.float().unsqueeze(0).unsqueeze(0), 2)[0, 0]) > 0.5
        gl2 = _grad_loss(res_d, mask_d)

        total = total + W_DEPTH_L1 * l_l1 + W_DEPTH_GRAD * (gl1 + gl2)

    n_valid = N - n_skipped
    if n_valid == 0:
        return torch.zeros(1, device=DEVICE)[0], n_skipped
    return total / n_valid, n_skipped


def psnr_in_mask(col_img, photo, mask):
    err_sq = (col_img.detach() - photo).pow(2).mean(-1)
    mse = (err_sq * mask.float()).sum() / mask.float().sum().clamp(min=1)
    return float(-10 * torch.log10(mse.clamp(min=1e-10)).item())


def cam_prior_loss(omega_t, t_cam_t):
    sig_r  = math.radians(SIG_R_DEG)
    r_pen  = omega_t[1:].pow(2).sum() / (sig_r ** 2)
    t_pen  = t_cam_t[1:].pow(2).sum() / (SIG_T ** 2)
    return W_CAM * (r_pen + t_pen) / max(N_TRAIN - 1, 1)


def wt_check(Fa):
    return check_watertight(Fa)


# ── Colour init ───────────────────────────────────────────────────────────────

def init_colours_from_photos(verts_t, faces_t, mvps_t, photos, res):
    V = verts_t.shape[0]
    ones_V = torch.ones(V, 1, device=DEVICE)
    vh = torch.cat([verts_t.detach(), ones_V], -1)
    color_acc = torch.zeros(V, 3, device=DEVICE)
    count_acc = torch.zeros(V, device=DEVICE)
    with torch.no_grad():
        for i in range(len(mvps_t)):
            pos_i = (mvps_t[i] @ vh.T).T.unsqueeze(0).contiguous()
            rast, _ = dr.rasterize(ctx, pos_i, faces_t, resolution=[res, res])
            fg = rast[0, :, :, 3] > 0
            if not fg.any():
                continue
            clip_v  = (mvps_t[i] @ vh.T).T
            w       = clip_v[:, 3].clamp(min=1e-6)
            x_ndc   = clip_v[:, 0] / w
            y_ndc   = clip_v[:, 1] / w
            ui      = ((x_ndc + 1) * 0.5 * res).long().clamp(0, res - 1)
            vi      = ((y_ndc + 1) * 0.5 * res).long().clamp(0, res - 1)
            in_bounds = (x_ndc.abs() <= 1) & (y_ndc.abs() <= 1)
            covered = rast[0, vi, ui, 3] > 0
            visible = in_bounds & covered
            if visible.any():
                p_col = photos[i, vi[visible], ui[visible], :]
                color_acc[visible] += p_col
                count_acc[visible] += 1
    C = (color_acc / count_acc.clamp(min=1).unsqueeze(-1)).clamp(0, 1)
    C[count_acc == 0] = 0.5
    return C


# ── Held-out eval ─────────────────────────────────────────────────────────────

@torch.no_grad()
def held_out_metrics(verts_t, faces_t, faces_l, C_t=None, T_t=None,
                     uvs_t_=None, uv_idx_t_=None, compute_depth_err=False):
    """Returns (ho_iou, ho_psnr, ho_depth_err or None).
    Supports vertex colour (C_t) OR texture (T_t + uvs_t_ + uv_idx_t_)."""
    N_ho = scene.ho_mvps.shape[0]
    N_v  = verts_t.shape[0]
    ones_V = torch.ones(N_v, 1, device=DEVICE)
    vh = torch.cat([verts_t, ones_V], -1)
    pos = (scene.ho_mvps @ vh.T).transpose(1, 2).contiguous()
    rast, rast_db = dr.rasterize(ctx, pos, faces_t, resolution=[res, res])

    ones_ho = torch.ones(N_ho, N_v, 3, device=DEVICE)
    sil_col, _ = dr.interpolate(ones_ho, rast, faces_t)
    sil  = dr.antialias(sil_col.contiguous(), rast, pos, faces_t)[..., 0]
    rast_fg = rast[..., 3] > 0

    # Appearance render
    if T_t is not None and uvs_t_ is not None and uv_idx_t_ is not None:
        uv_exp = uvs_t_.unsqueeze(0).expand(N_ho, -1, -1).contiguous()
        uv_img, uv_da = dr.interpolate(uv_exp, rast, uv_idx_t_,
                                        rast_db=rast_db, diff_attrs='all')
        col_img = dr.texture(T_t, uv_img, uv_da,
                             filter_mode='linear-mipmap-linear')
        col_img = dr.antialias(col_img.contiguous(), rast, pos, faces_t)
    elif C_t is not None:
        C_exp = C_t.unsqueeze(0).expand(N_ho, -1, -1).contiguous()
        col_img, _ = dr.interpolate(C_exp, rast, faces_t)
        col_img = dr.antialias(col_img.contiguous(), rast, pos, faces_t)
    else:
        # fallback: grey
        C_grey = torch.full((N_v, 3), 0.5, device=DEVICE)
        C_exp = C_grey.unsqueeze(0).expand(N_ho, -1, -1).contiguous()
        col_img, _ = dr.interpolate(C_exp, rast, faces_t)
        col_img = dr.antialias(col_img.contiguous(), rast, pos, faces_t)

    if SHADE:
        R     = scene.ho_mvps[:, :3, :3]
        vn    = compute_vertex_normals_grad(verts_t, faces_l)
        n_cam = torch.bmm(R, vn.unsqueeze(0).expand(N_ho,-1,-1).transpose(1,2)).transpose(1,2)
        n_cam = (n_cam / n_cam.norm(dim=-1,keepdim=True).clamp(min=1e-8)).contiguous()
        n_img,_ = dr.interpolate(n_cam, rast, faces_t)
        n_img = n_img / n_img.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        shade = AMB + (1-AMB) * n_img[..., 2].clamp(min=0)
        col_img = (col_img * shade.unsqueeze(-1)).clamp(0, 1)

    pred_fg = sil > 0.5
    inter  = (pred_fg & ho_gt_t).float().sum()
    union  = (pred_fg | ho_gt_t).float().sum().clamp(min=1)
    ho_iou = float(inter / union)
    ho_psnr = psnr_in_mask(col_img, ho_photo, ho_interior & rast_fg)

    # Depth error on held-out views
    ho_depth_err = None
    if compute_depth_err and HAS_DEPTH:
        vh_d = torch.cat([verts_t, torch.ones(N_v, 1, device=DEVICE)], -1)
        z_cam     = -(scene.ho_view_w2c @ vh_d.T)[:, 2, :]
        inv_z_v   = (1.0 / z_cam.clamp(min=1e-6)).unsqueeze(-1)
        inv_d_img,_ = dr.interpolate(inv_z_v.contiguous(), rast, faces_t)
        q_ho      = inv_d_img[..., 0]
        total_err = 0.0; n_good = 0
        for i in range(N_ho):
            mask = rast_fg[i] & ho_interior[i]
            n_px = int(mask.float().sum().item())
            if n_px < 200:
                continue
            q_m  = q_ho[i][mask].detach()
            m_m  = ho_mono[i][mask]
            m_c  = m_m - m_m.mean(); q_c = q_m - q_m.mean()
            dnom = (m_c*m_c).sum().clamp(min=1e-8)
            s    = (m_c*q_c).sum() / dnom
            b    = q_m.mean() - s * m_m.mean()
            if float(s.item()) <= 0:
                continue
            err = ((s * m_m + b - q_m).abs() / q_m.abs().mean().clamp(min=1e-8)).mean()
            total_err += float(err.item()); n_good += 1
        ho_depth_err = total_err / max(n_good, 1)

    return ho_iou, ho_psnr, ho_depth_err


# ═══════════════════════════════════════════════════════════════════════════════
# Main training loop
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    verts_t = torch.tensor(V0, dtype=torch.float32, device=DEVICE).requires_grad_(True)
    faces_t = torch.tensor(F0, dtype=torch.int32,   device=DEVICE)
    faces_l = faces_t.long()

    crump_input = crumpling_metric(V0, F0)
    print(f"[p8] input crumpling: {crump_input:.3f}°", flush=True)

    src, dst, deg, _ = build_adj(F0, len(V0), want_excl=False)
    pairs_t = torch.tensor(build_pairs(F0), device=DEVICE)

    # ── Appearance init ──────────────────────────────────────────────────────
    if TEX:
        # Texture already created (T_tex, global)
        C_t = None
        print(f"[p8] TEX=1: using UV texture [{TEX_RES}x{TEX_RES}]", flush=True)
    else:
        print("[p8] initialising vertex colours ...", flush=True)
        with torch.no_grad():
            C_init = init_colours_from_photos(verts_t, faces_t, scene.mvps, photo_t, res)
        C_t = C_init.clone().requires_grad_(True)
        print(f"[p8] colour init: mean={C_init.mean():.3f} "
              f"covered={int((C_init!=0.5).any(-1).float().sum())}/{len(V0)}", flush=True)

    # ── Camera perturbation (v3) ─────────────────────────────────────────────
    omega_perturb_t = None
    if CAM_PERTURB_DEG > 0:
        rng = np.random.RandomState(CAM_PERTURB_SEED)
        axes = rng.randn(N_TRAIN, 3).astype(np.float32)
        axes /= np.linalg.norm(axes, axis=1, keepdims=True) + 1e-12
        angles = rng.randn(N_TRAIN).astype(np.float32) * math.radians(CAM_PERTURB_DEG)
        angles[0] = 0  # camera 0 untouched
        omega_p = (axes * angles[:, None]).astype(np.float32)
        omega_p[0] = 0
        omega_perturb_t = torch.tensor(omega_p, device=DEVICE)
        delta_p = build_delta(omega_perturb_t, torch.zeros(N_TRAIN, 3, device=DEVICE))
        with torch.no_grad():
            scene.view_w2c = torch.bmm(delta_p, scene.view_w2c)
            scene.mvps = torch.bmm(scene.proj, scene.view_w2c)
        mean_pert = float(omega_perturb_t[1:].norm(dim=-1).mean() * 180 / math.pi)
        print(f"[p8] CAM_PERTURB applied: mean={mean_pert:.3f}° seed={CAM_PERTURB_SEED}", flush=True)

    omega = torch.zeros(N_TRAIN, 3, device=DEVICE)
    t_cam = torch.zeros(N_TRAIN, 3, device=DEVICE)
    if JOINT_CAM:
        omega = omega.requires_grad_(True)
        t_cam = t_cam.requires_grad_(True)

    # ── Phase A: colour only (TEX=0 + FREEZE_GEO=0 only) ────────────────────
    ho_iou_a = ho_psnr_a = ho_derr_a = None
    skip_phase_a = TEX or FREEZE_GEO
    if not skip_phase_a:
        opt_A   = torch.optim.Adam([C_t], lr=LR_C * 2)
        sched_A = torch.optim.lr_scheduler.CosineAnnealingLR(opt_A, T_max=STEPS_A,
                                                              eta_min=LR_C * 0.1)
        print(f"\n[p8] === Phase A: colour only, {STEPS_A} steps ===", flush=True)
        psnr_a_init = None
        for step in range(STEPS_A):
            opt_A.zero_grad()
            sil, col_out, col_raw, rast_fg, _ = render_batch(
                verts_t.detach(), faces_t, faces_l, scene.mvps,
                C_t=C_t.clamp(0, 1),
                view_w2c_eff=scene.view_w2c, do_shade=True, do_depth=False)
            sl = _bl.sil_loss_batch(sil, gt_soft)
            pl = pho_loss_fn(col_out, rast_fg, gt_interior, valid_t)
            cl = colour_lap_loss(C_t, src, dst, deg)
            loss = W_SIL * sl + W_PHO * pl + W_CLAP * cl
            loss.backward()
            opt_A.step(); sched_A.step()
            if step == 0:
                with torch.no_grad():
                    _, psnr_a_init, _ = held_out_metrics(verts_t, faces_t, faces_l,
                                                          C_t=C_t.detach().clamp(0, 1))
                print(f"[p8] Phase A init: ho_PSNR={psnr_a_init:.2f} dB", flush=True)
            if (step + 1) % 100 == 0:
                with torch.no_grad():
                    ho_iou, ho_psnr, _ = held_out_metrics(verts_t, faces_t, faces_l,
                                                            C_t=C_t.detach().clamp(0, 1))
                print(f"[p8] A step {step+1:4d}: sl={sl.item():.4f} pho={pl.item():.4f} "
                      f"clap={cl.item():.4f} | ho_IoU={ho_iou:.4f} ho_PSNR={ho_psnr:.2f}dB",
                      flush=True)
        with torch.no_grad():
            ho_iou_a, ho_psnr_a, ho_derr_a = held_out_metrics(
                verts_t, faces_t, faces_l, C_t=C_t.detach().clamp(0, 1),
                compute_depth_err=HAS_DEPTH)
        print(f"[p8] Phase A final: ho_IoU={ho_iou_a:.4f} ho_PSNR={ho_psnr_a:.2f}dB", flush=True)
        if psnr_a_init is not None and not (ho_psnr_a > psnr_a_init):
            print(f"[p8] WARNING: Phase A PSNR did not improve", flush=True)
    else:
        print(f"\n[p8] Phase A skipped (TEX={TEX} FREEZE_GEO={FREEZE_GEO})", flush=True)

    # ── Phase B: main optimisation ────────────────────────────────────────────
    param_groups = []
    _cam_pg_idx = []
    if not FREEZE_GEO:
        param_groups.append({'params': [verts_t], 'lr': LR_V})
    if TEX:
        param_groups.append({'params': [T_tex], 'lr': LR_T})
    else:
        param_groups.append({'params': [C_t], 'lr': LR_C})
    if JOINT_CAM:
        _cam_lr = LR_CAM_INIT if LR_CAM_INIT > 0 else LR_CAM
        _cam_pg_idx = [len(param_groups), len(param_groups) + 1]
        param_groups += [{'params': [omega], 'lr': _cam_lr},
                         {'params': [t_cam], 'lr': _cam_lr}]

    opt_B = torch.optim.Adam(param_groups)
    def _cosine_decay(step):
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * step / max(STEPS_B, 1))))
    sched_B = torch.optim.lr_scheduler.LambdaLR(opt_B, lr_lambda=_cosine_decay)

    _cam_stepped_down = False
    print(f"\n[p8] === Phase B: {STEPS_B} steps (JOINT_CAM={JOINT_CAM} "
          f"FREEZE_GEO={FREEZE_GEO} TEX={TEX}) ===", flush=True)

    for step in range(STEPS_B):
        opt_B.zero_grad()

        # Camera LR step-down at 60% (v3 recovery test)
        if (JOINT_CAM and LR_CAM_INIT > 0 and not _cam_stepped_down
                and step >= int(0.6 * STEPS_B)):
            for idx in _cam_pg_idx:
                opt_B.param_groups[idx]['lr'] = LR_CAM
            _cam_stepped_down = True
            print(f"[p8] B step {step}: camera LR stepped down to {LR_CAM}", flush=True)

        if JOINT_CAM:
            mvps_eff   = eff_mvps(omega, t_cam)
            view_w2c_b = None
        else:
            mvps_eff   = scene.mvps
            view_w2c_b = scene.view_w2c

        sil, col_out, col_raw, rast_fg, inv_depth = render_batch(
            verts_t if not FREEZE_GEO else verts_t.detach(),
            faces_t, faces_l, mvps_eff,
            C_t=C_t.clamp(0, 1) if (C_t is not None and not TEX) else None,
            T_t=T_tex if TEX else None,
            uvs_t_=uvs_t if TEX else None,
            uv_idx_t_=uv_idx_t if TEX else None,
            mip_bias_val=mip_bias_schedule(step, STEPS_B) if TEX else 0.0,
            view_w2c_eff=view_w2c_b,
            do_shade=True, do_depth=HAS_DEPTH and not FREEZE_GEO)

        # ── Losses ──
        sl = _bl.sil_loss_batch(sil, gt_soft)

        if not FREEZE_GEO:
            me   = mean_edge_of(verts_t.detach(), src, dst)
            ramp = min(1.0, max(0.0, (step - 200) / 400.0))
            geo  = (W_LAP_J  * laplacian_loss(verts_t, faces_t)
                  + W_EDGE_J * edge_length_loss(verts_t, faces_t)
                  + W_SPIKE  * spike_pen(verts_t, src, dst, deg, me)
                  + W_SLIVER * sliver_pen(verts_t, faces_l, me)
                  + W_T      * field_dist(verts_t).mean()
                  + W_NC     * normal_consistency_loss(verts_t, faces_l, pairs_t))
            fold_l = W_FOLD * ramp * fold_loss(verts_t, faces_l, pairs_t)
        else:
            geo = fold_l = torch.zeros(1, device=DEVICE)[0]

        pl = pho_loss_fn(col_out, rast_fg, gt_interior, valid_t)

        if TEX:
            tv_l = W_TV * texture_tv_loss(T_tex)
            cl = torch.zeros(1, device=DEVICE)[0]
        else:
            cl = colour_lap_loss(C_t, src, dst, deg) if C_t is not None else torch.zeros(1, device=DEVICE)[0]
            tv_l = torch.zeros(1, device=DEVICE)[0]

        cam_l = cam_prior_loss(omega, t_cam) if JOINT_CAM else torch.zeros(1, device=DEVICE)[0]

        if HAS_DEPTH and inv_depth is not None and not FREEZE_GEO:
            dl, n_skip = depth_mono_loss_fn(inv_depth, rast_fg, gt_interior, valid_t, mono_t)
            depth_loss_val = W_DEPTH_MONO * dl
        else:
            depth_loss_val = torch.zeros(1, device=DEVICE)[0]; n_skip = 0

        loss = (W_SIL * sl + W_PHO * pl + W_CLAP * cl + tv_l
                + geo + fold_l + cam_l + depth_loss_val)
        loss.backward()

        if JOINT_CAM:
            with torch.no_grad():
                if omega.grad is not None: omega.grad[0] = 0
                if t_cam.grad is not None: t_cam.grad[0] = 0

        opt_B.step(); sched_B.step()
        with torch.no_grad():
            if C_t is not None and not TEX: C_t.clamp_(0, 1)
            if TEX: T_tex.data.clamp_(0, 1)

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                ho_iou, ho_psnr, _ = held_out_metrics(
                    verts_t, faces_t, faces_l,
                    C_t=C_t.clamp(0, 1) if (C_t is not None and not TEX) else None,
                    T_t=T_tex if TEX else None, uvs_t_=uvs_t, uv_idx_t_=uv_idx_t)
                tr_psnr = psnr_in_mask(col_out.detach(), photo_t, rast_fg & gt_interior)
                om_deg  = float(omega.detach().norm(dim=-1).mean() * 180/math.pi) if JOINT_CAM else 0.0
                t_norm  = float(t_cam.detach().norm(dim=-1).mean()) if JOINT_CAM else 0.0
            mb = mip_bias_schedule(step, STEPS_B) if TEX else 0.0
            print(f"[p8] B step {step+1:5d}: sl={sl.item():.4f} pho={pl.item():.4f} "
                  f"dep={depth_loss_val.item():.4f} tv={tv_l.item():.4f} "
                  f"tr_PSNR={tr_psnr:.2f} | ho_IoU={ho_iou:.4f} ho_PSNR={ho_psnr:.2f}dB "
                  f"ω={om_deg:.3f}° |t|={t_norm:.4f} mip={mb:.1f}", flush=True)

        if (step + 1) % 250 == 0:
            ok, nbad = wt_check(F0)
            print(f"[p8] B step {step+1}: watertight={ok} bad_edges={nbad}", flush=True)

    # ── Final metrics ─────────────────────────────────────────────────────────
    with torch.no_grad():
        ho_iou_b, ho_psnr_b, ho_derr_b = held_out_metrics(
            verts_t, faces_t, faces_l,
            C_t=C_t.clamp(0, 1) if (C_t is not None and not TEX) else None,
            T_t=T_tex if TEX else None, uvs_t_=uvs_t, uv_idx_t_=uv_idx_t,
            compute_depth_err=HAS_DEPTH)
    print(f"[p8] Phase B final: ho_IoU={ho_iou_b:.4f} ho_PSNR={ho_psnr_b:.2f}dB "
          f"ho_derr={ho_derr_b}", flush=True)
    wt_ok, _ = wt_check(F0)
    print(f"[p8] watertight: {wt_ok}", flush=True)

    V_final = verts_t.detach().cpu().numpy()
    crump_final = crumpling_metric(V_final, F0)
    print(f"[p8] final crumpling: {crump_final:.3f}° (input {crump_input:.3f}°, "
          f"ratio {crump_final/max(crump_input,1e-4):.2f}x)", flush=True)

    # ── Camera recovery report (v3) ──────────────────────────────────────────
    if CAM_PERTURB_DEG > 0 and omega_perturb_t is not None and JOINT_CAM:
        with torch.no_grad():
            residual = (omega.detach() + omega_perturb_t)[1:].norm(dim=-1)
            perturbation = omega_perturb_t[1:].norm(dim=-1)
            mean_res_deg = float(residual.mean() * 180 / math.pi)
            max_res_deg  = float(residual.max() * 180 / math.pi)
            mean_pert_deg = float(perturbation.mean() * 180 / math.pi)
            recovery_pct = (1.0 - mean_res_deg / max(mean_pert_deg, 1e-8)) * 100
        print(f"\n[p8] RECOVERY: mean_residual={mean_res_deg:.4f}° "
              f"max_residual={max_res_deg:.4f}° "
              f"perturbation={mean_pert_deg:.4f}° recovery={recovery_pct:.1f}%", flush=True)
    elif CAM_PERTURB_DEG == 0 and JOINT_CAM:
        # Control: report drift from COLMAP
        with torch.no_grad():
            om_drift = float(omega.detach()[1:].norm(dim=-1).mean() * 180 / math.pi)
            t_drift  = float(t_cam.detach()[1:].norm(dim=-1).mean())
        print(f"\n[p8] CONTROL drift from COLMAP: mean_ω={om_drift:.4f}° mean_|t|={t_drift:.6f}",
              flush=True)

    # ── Pose-consistency metric (v3, for cam runs) ───────────────────────────
    if JOINT_CAM and not FREEZE_GEO:
        with torch.no_grad():
            mvps_for_pc = eff_mvps(omega.detach(), t_cam.detach())
        lo, hi = HF.lo, HF.hi
        pc_med, pc_all = pose_consistency_metric(mvps_for_pc, gt_np, lo, hi)
        print(f"[p8] pose-consistency (corrected): median={pc_med:.4f}", flush=True)
        # Also compute baseline (COLMAP)
        pc_med_col, _ = pose_consistency_metric(scene.mvps, gt_np, HF.lo, HF.hi)
        print(f"[p8] pose-consistency (COLMAP baseline): median={pc_med_col:.4f}", flush=True)

    # ── Input mesh baseline ───────────────────────────────────────────────────
    with torch.no_grad():
        verts_init = torch.tensor(V0, dtype=torch.float32, device=DEVICE)
        C_grey = torch.full((len(V0), 3), 0.5, device=DEVICE)
        ho_iou_0, ho_psnr_0, ho_derr_0 = held_out_metrics(
            verts_init, faces_t, faces_l, C_t=C_grey, compute_depth_err=HAS_DEPTH)

    # ── Save outputs ──────────────────────────────────────────────────────────
    out_npz = f"despike/results_genus/{SHAPE}_joint_{TAGP}.npz"
    save_kw = dict(verts=V_final, tris=F0)
    if TEX:
        save_kw['uvs'] = _uvs.astype(np.float32)
        save_kw['uv_idx'] = _uv_indices.astype(np.int32)
    else:
        C_final = C_t.detach().clamp(0, 1).cpu().numpy() if C_t is not None else np.full((len(V0), 3), 0.5, dtype=np.float32)
        save_kw['colors'] = C_final
    np.savez(out_npz, **save_kw)
    print(f"[p8] saved: {out_npz}", flush=True)

    # Save texture PNG (v3)
    if TEX:
        import cv2
        tex_np = T_tex.detach().clamp(0, 1)[0].cpu().numpy()  # [H,W,3]
        tex_u8 = (tex_np * 255).clip(0, 255).astype(np.uint8)
        tex_path = f"{REAL_DATA}/tex_{TAGP}.png"
        cv2.imwrite(tex_path, cv2.cvtColor(tex_u8, cv2.COLOR_RGB2BGR))
        print(f"[p8] texture PNG: {tex_path}", flush=True)

    # Camera JSON
    if JOINT_CAM:
        cam_data = {f"train_{i:03d}": {"omega": omega.detach()[i].tolist(),
                                       "t": t_cam.detach()[i].tolist()}
                    for i in range(N_TRAIN)}
        cam_data["summary"] = {
            "mean_omega_deg": float(omega.detach().norm(dim=-1).mean() * 180/math.pi),
            "mean_t_norm":    float(t_cam.detach().norm(dim=-1).mean()),
        }
        if omega_perturb_t is not None:
            cam_data["recovery"] = {
                "mean_residual_deg": mean_res_deg if CAM_PERTURB_DEG > 0 else None,
                "recovery_pct": recovery_pct if CAM_PERTURB_DEG > 0 else None,
            }
    else:
        cam_data = {"summary": {"note": "JOINT_CAM=0, cameras frozen"}}
    cam_json = f"{REAL_DATA}/cams_joint_{TAGP}.json"
    with open(cam_json, "w") as fh: json.dump(cam_data, fh, indent=2)

    # ── Views PNG: 4 held-out × [photo | appearance render | geometry shaded] ──
    views_png = f"{REAL_DATA}/joint_{TAGP}_views.png"
    _make_views_png_v3(verts_t.detach(), faces_t, faces_l,
                       C_t.detach().clamp(0,1) if (C_t is not None and not TEX) else None,
                       T_tex if TEX else None, uvs_t, uv_idx_t, views_png)

    # ── Geometry-only turntable (--no-colors) ─────────────────────────────────
    geo_png = f"{REAL_DATA}/joint_{TAGP}_geo_grid.png"
    _make_geo_grid(verts_t.detach(), faces_t, faces_l, geo_png)
    turn_out = f"{REAL_DATA}/joint_{TAGP}"
    _make_turn_mp4_geo(out_npz, turn_out)

    # ── Markdown table ────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n[p8] Wall time: {elapsed/60:.1f} min", flush=True)
    print(f"\n## Phase8 v3 Results (TAGP={TAGP})\n")
    print(f"| Metric | Value |")
    print(f"|--------|-------|")
    print(f"| ho sil IoU | {ho_iou_b:.4f} |")
    print(f"| ho PSNR (dB) | {ho_psnr_b:.2f} |")
    if ho_derr_b is not None: print(f"| ho depth err | {ho_derr_b:.4f} |")
    print(f"| crumpling (°) | {crump_final:.3f} (input {crump_input:.3f}) |")
    print(f"| watertight | {wt_ok} |")
    om_fin = float(omega.detach().norm(dim=-1).mean()*180/math.pi) if JOINT_CAM else 0.0
    t_fin  = float(t_cam.detach().norm(dim=-1).mean()) if JOINT_CAM else 0.0
    print(f"| mean ω (°) | {om_fin:.4f} |")
    print(f"| mean |t| | {t_fin:.6f} |")
    if CAM_PERTURB_DEG > 0 and JOINT_CAM:
        print(f"| recovery % | {recovery_pct:.1f} |")
    print(f"| wall (min) | {elapsed/60:.1f} |")


# ═══════════════════════════════════════════════════════════════════════════════
# Output helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _render_ho_view(verts_t, faces_t, faces_l, C_t, pick, geo_only=False, depth_out=False):
    """Render a single held-out view (vertex colour). Returns (col_np_256, geo_np_256, None)."""
    N_v = verts_t.shape[0]
    ones_V = torch.ones(N_v, 1, device=DEVICE)
    vh = torch.cat([verts_t, ones_V], -1)
    mvp = scene.ho_mvps[pick]
    pos = (mvp @ vh.T).T.unsqueeze(0).contiguous()
    rast, _ = dr.rasterize(ctx, pos, faces_t, resolution=[256, 256])
    fg = (rast[0, :, :, 3] > 0).float().unsqueeze(-1)
    C_exp = C_t.unsqueeze(0).contiguous()
    col, _ = dr.interpolate(C_exp, rast, faces_t)
    col    = dr.antialias(col.contiguous(), rast, pos, faces_t)[0]
    col_np = ((col * fg + torch.ones_like(col) * (1-fg)).flip(0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    # Geometry-only shaded
    grey = torch.ones(N_v, 3, device=DEVICE) * 0.7
    g_exp = grey.unsqueeze(0).contiguous()
    g_col,_ = dr.interpolate(g_exp, rast, faces_t)
    g_col   = dr.antialias(g_col.contiguous(), rast, pos, faces_t)[0]
    R    = mvp[:3, :3].unsqueeze(0)
    vn   = compute_vertex_normals_grad(verts_t, faces_l)
    n_c  = torch.bmm(R, vn.unsqueeze(0).transpose(1,2)).transpose(1,2)
    n_c  = (n_c / n_c.norm(dim=-1,keepdim=True).clamp(min=1e-8)).contiguous()
    n_img,_ = dr.interpolate(n_c, rast, faces_t)
    n_img   = n_img / n_img.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    shade   = AMB + (1-AMB) * n_img[0,:,:,2].clamp(min=0)
    g_col   = (g_col * shade.unsqueeze(-1)).clamp(0,1)
    geo_np = ((g_col * fg + torch.ones_like(g_col)*(1-fg)).flip(0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    return col_np, geo_np, None


def _render_ho_view_v3(verts_t, faces_t, faces_l, pick, C_t=None,
                        T_t=None, uvs_t_=None, uv_idx_t_=None, size=256):
    """Render held-out view: appearance (texture or vertex colour) + geometry-only shaded.
    Returns (appearance_np, geo_shaded_np) both [size,size,3] uint8 RGB."""
    N_v = verts_t.shape[0]
    ones_V = torch.ones(N_v, 1, device=DEVICE)
    vh = torch.cat([verts_t, ones_V], -1)
    mvp = scene.ho_mvps[pick]
    pos = (mvp @ vh.T).T.unsqueeze(0).contiguous()
    rast, rast_db = dr.rasterize(ctx, pos, faces_t, resolution=[size, size])
    fg = (rast[0, :, :, 3] > 0).float().unsqueeze(-1)
    # Appearance render
    if T_t is not None and uvs_t_ is not None and uv_idx_t_ is not None:
        uv_exp = uvs_t_.unsqueeze(0).contiguous()
        uv_img, uv_da = dr.interpolate(uv_exp, rast, uv_idx_t_,
                                        rast_db=rast_db, diff_attrs='all')
        col = dr.texture(T_t, uv_img, uv_da, filter_mode='linear-mipmap-linear')
        col = dr.antialias(col.contiguous(), rast, pos, faces_t)[0]
    elif C_t is not None:
        C_exp = C_t.unsqueeze(0).contiguous()
        col, _ = dr.interpolate(C_exp, rast, faces_t)
        col = dr.antialias(col.contiguous(), rast, pos, faces_t)[0]
    else:
        col = torch.ones(size, size, 3, device=DEVICE) * 0.5
    app_np = ((col * fg + torch.ones_like(col) * (1-fg)).flip(0).cpu().numpy() * 255).clip(0,255).astype(np.uint8)
    # Geometry-only shaded (ALWAYS shaded — spec: NOT a flat silhouette)
    grey = torch.ones(N_v, 3, device=DEVICE) * 0.7
    g_exp = grey.unsqueeze(0).contiguous()
    g_col, _ = dr.interpolate(g_exp, rast, faces_t)
    g_col = dr.antialias(g_col.contiguous(), rast, pos, faces_t)[0]
    R   = mvp[:3, :3].unsqueeze(0)
    vn  = compute_vertex_normals_grad(verts_t, faces_l)
    n_c = torch.bmm(R, vn.unsqueeze(0).transpose(1,2)).transpose(1,2)
    n_c = (n_c / n_c.norm(dim=-1,keepdim=True).clamp(min=1e-8)).contiguous()
    n_img, _ = dr.interpolate(n_c, rast, faces_t)
    n_img = n_img / n_img.norm(dim=-1,keepdim=True).clamp(min=1e-8)
    shade = AMB + (1-AMB) * n_img[0,:,:,2].clamp(min=0)
    g_col = (g_col * shade.unsqueeze(-1)).clamp(0,1)
    geo_np = ((g_col * fg + torch.ones_like(g_col)*(1-fg)).flip(0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
    return app_np, geo_np


def _make_views_png_v3(verts_t, faces_t, faces_l, C_t, T_t, uvs_t_, uv_idx_t_, out_path):
    """4 held-out views x [photo | appearance render | geometry shaded]."""
    import cv2
    picks = [0, 4, 8, 12]
    rows = []
    with torch.no_grad():
        for pick in picks:
            photo_np = (scene.ho_rgb[pick] * 255).astype(np.uint8)
            photo_np = np.flip(photo_np, 0).copy()
            ph_show = cv2.resize(photo_np, (256, 256))
            app_np, geo_np = _render_ho_view_v3(
                verts_t, faces_t, faces_l, pick,
                C_t=C_t, T_t=T_t, uvs_t_=uvs_t_, uv_idx_t_=uv_idx_t_, size=256)
            row = np.concatenate([ph_show, app_np, geo_np], axis=1)
            rows.append(row)
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[p8] views PNG (v3): {out_path}", flush=True)


def _make_views_png(verts_t, faces_t, faces_l, C_t, out_path):
    """Legacy v1/v2: 4 held-out views x [photo | render | geo]."""
    import cv2
    picks = [0, 4, 8, 12]
    rows = []
    with torch.no_grad():
        for pick in picks:
            photo_np = (scene.ho_rgb[pick] * 255).astype(np.uint8)
            photo_np = np.flip(photo_np, 0).copy()
            ph_show  = cv2.resize(photo_np, (256, 256))
            col_np, geo_np, _ = _render_ho_view(verts_t, faces_t, faces_l, C_t, pick)
            row = np.concatenate([ph_show, col_np, geo_np], axis=1)
            rows.append(row)
    grid = np.concatenate(rows, axis=0)
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[p8] views PNG: {out_path}", flush=True)


def _make_geo_grid(verts_t, faces_t, faces_l, out_path):
    """4 held-out views, geometry-only shaded renders."""
    import cv2
    picks = [0, 4, 8, 12]
    cols = []
    with torch.no_grad():
        for pick in picks:
            _, geo_np, _ = _render_ho_view(verts_t, faces_t, faces_l,
                                           torch.zeros(verts_t.shape[0], 3, device=DEVICE),
                                           pick, geo_only=True, depth_out=False)
            cols.append(geo_np)
    grid = np.concatenate(cols, axis=1)
    cv2.imwrite(out_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    print(f"[p8] geo grid PNG: {out_path}", flush=True)


def _make_turn_mp4_geo(npz_path, out_prefix):
    """Turntable with geometry-only (--no-colors)."""
    cmd = ["python3", "real/render_mesh.py", npz_path, out_prefix,
           "--turn", "72", "--res", "512", "--no-colors"]
    try:
        subprocess.run(cmd, check=True,
                       cwd="/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
        print(f"[p8] turntable: {out_prefix}_turn.mp4", flush=True)
    except Exception as e:
        print(f"[p8] WARNING: turntable failed: {e}", flush=True)


def _render_geo_views_npz(npz_path, picks=(0, 4, 8, 12), size=256):
    """Render geometry-only shaded views for a mesh npz at held-out camera positions."""
    z   = np.load(npz_path)
    V   = z["verts"].astype(np.float32)
    F_  = z["tris"].astype(np.int32)
    vt  = torch.tensor(V, device=DEVICE)
    ft  = torch.tensor(F_, dtype=torch.int32, device=DEVICE)
    fl  = ft.long()
    N_v = vt.shape[0]
    images = []
    ones_V = torch.ones(N_v, 1, device=DEVICE)
    vh = torch.cat([vt, ones_V], -1)
    with torch.no_grad():
        for pick in picks:
            mvp = scene.ho_mvps[pick]
            pos = (mvp @ vh.T).T.unsqueeze(0).contiguous()
            rast, _ = dr.rasterize(ctx, pos, ft, resolution=[size, size])
            fg = (rast[0,:,:,3] > 0).float().unsqueeze(-1)
            grey = torch.ones(N_v, 3, device=DEVICE) * 0.7
            g_exp = grey.unsqueeze(0).contiguous()
            g_col,_ = dr.interpolate(g_exp, rast, ft)
            g_col   = dr.antialias(g_col.contiguous(), rast, pos, ft)[0]
            vn  = compute_vertex_normals_grad(vt, fl)
            R   = mvp[:3, :3].unsqueeze(0)
            n_c = torch.bmm(R, vn.unsqueeze(0).transpose(1,2)).transpose(1,2)
            n_c = (n_c / n_c.norm(dim=-1,keepdim=True).clamp(min=1e-8)).contiguous()
            n_img,_ = dr.interpolate(n_c, rast, ft)
            n_img   = n_img / n_img.norm(dim=-1,keepdim=True).clamp(min=1e-8)
            shade   = AMB + (1-AMB) * n_img[0,:,:,2].clamp(min=0)
            g_col   = (g_col * shade.unsqueeze(-1)).clamp(0,1)
            np_img  = ((g_col * fg + torch.ones_like(g_col)*(1-fg)).flip(0).cpu().numpy()*255).clip(0,255).astype(np.uint8)
            images.append(np_img)
    return images


def make_comparison_sheet():
    """One sheet: rows=[real7, cam1, d1s0, d1s1, d0s1], cols=4 geo views."""
    import cv2
    picks = [0, 4, 8, 12]
    meshes = [
        ("real7_input",  f"despike/results_genus/dino_real7_auto.npz"),
        ("v1_cam1",      f"despike/results_genus/dino_joint_cam1.npz"),
        ("d1s0",         f"despike/results_genus/dino_joint_d1s0.npz"),
        ("d1s1",         f"despike/results_genus/dino_joint_d1s1.npz"),
        ("d0s1",         f"despike/results_genus/dino_joint_d0s1.npz"),
    ]
    rows = []
    for label, npz in meshes:
        if not os.path.exists(npz):
            print(f"[p8] comparison sheet: missing {npz}, skipping row", flush=True)
            continue
        imgs = _render_geo_views_npz(npz, picks=picks, size=256)
        row  = np.concatenate(imgs, axis=1)    # [256, 4*256, 3]
        # Add row label
        cv2.putText(row, label, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2)
        rows.append(row)
    if not rows:
        print("[p8] comparison sheet: no meshes found", flush=True)
        return
    sheet = np.concatenate(rows, axis=0)
    out   = f"{REAL_DATA}/cmp_joint_v2.png"
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print(f"[p8] comparison sheet: {out}", flush=True)


if __name__ == "__main__":
    main()
