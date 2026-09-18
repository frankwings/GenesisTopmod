"""real_scene.py — real-data adapter for the golden topo-carving chain.

Usage (acceptance test 1):
  python3 despike/real_scene.py real/dino3

Full chain usage:
  REAL_DATA=real/dino3 SHAPE=dino TAGP=real1 bash golden_v3_chain.sh

Convention note: nvdiffrast stores rasterized output with row 0 at the BOTTOM
(NDC y = -1), matching OpenGL convention.  PIL/numpy images have row 0 at the
TOP. All masks are flipped vertically on load so they share the nvdiffrast
convention; self.gt and ho_gt are also stored flipped.  The projection matrix
uses flip_y=False (standard OpenGL direction), and the empirical probe only
tests flip_x (since y is handled by the mask flip).
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import json, math, time
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import binary_dilation, label as cc_label, binary_fill_holes

from cow_v13 import DEVICE

TRAIN_RES = int(os.environ.get("TRAIN_RES", "256"))


# ─────────────────────────────────────────────────────────────────────────────
# Projection matrix from COLMAP intrinsics (for a res×res cropped image)
# ─────────────────────────────────────────────────────────────────────────────

def _proj_matrix(f_pix, cx_crop, cy_crop, res, near, far, flip_x=False):
    """Build 4×4 OpenGL projection from pinhole intrinsics for a res×res image.

    Standard (flip_y=False in the sense that Y_gl>0 → y_ndc>0 → upper half).
    Masks must be stored in nvdiffrast convention (row 0 = bottom) for carve_hull.

    x_ndc = (2f/res)*X_gl/W + (1 - 2*cx/res)   where W = -Z_gl
    y_ndc = (2f/res)*Y_gl/W + (2*cy/res - 1)
    flip_x: negate x_ndc (test if COLMAP x-axis is mirrored vs OpenGL).
    """
    P = np.zeros((4, 4), np.float32)
    P[0, 0] =  2.0 * f_pix / res
    P[0, 2] =  1.0 - 2.0 * cx_crop / res
    P[1, 1] =  2.0 * f_pix / res
    P[1, 2] =  2.0 * cy_crop / res - 1.0
    P[2, 2] = (far + near) / (near - far)
    P[2, 3] =  2.0 * far * near / (near - far)
    P[3, 2] = -1.0
    if flip_x:
        P[0, 0] = -P[0, 0]
        P[0, 2] = -P[0, 2]
    return P


def _view_matrix_norm(R, t, X, s):
    """Build 4×4 view matrix (world_norm → OpenGL camera) for the NORMALISED world.

    Normalised world: V_norm = (V_world - X) / s.
    COLMAP: V_cam = R @ V_world + t → V_cam = s*R @ V_norm + (R@X + t).
    OpenGL convention (x right, y up, z backward = flip y and z vs COLMAP):
      row 0: +s*R[0]  (x right, same)
      row 1: -s*R[1]  (y up, was y down)
      row 2: -s*R[2]  (z backward, was z forward)
    Translation column: [t_eff[0], -t_eff[1], -t_eff[2]] where t_eff = R@X + t.
    """
    t_eff = R @ X + t
    M = np.eye(4, dtype=np.float32)
    M[0, :3] =  s * R[0]
    M[1, :3] = -s * R[1]
    M[2, :3] = -s * R[2]
    M[0, 3]  =  t_eff[0]
    M[1, 3]  = -t_eff[1]
    M[2, 3]  = -t_eff[2]
    return M


# ─────────────────────────────────────────────────────────────────────────────
# World normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _ray_intersection(rays_d, rays_o):
    """Least-squares intersection: A X = b  with A = Σ(I-d d^T), b = Σ(I-d d^T)o."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for d, o in zip(rays_d, rays_o):
        d = d / (np.linalg.norm(d) + 1e-12)
        P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def _mask_centroid_ray(mask_hw, R, t, f, cx_full, cy_full):
    """Back-project mask centroid (PIL convention: row 0 = top) to world ray."""
    ys, xs = np.where(mask_hw > 127)
    if len(ys) == 0:
        return None, None
    cy_pix = float(ys.mean()); cx_pix = float(xs.mean())
    xc = (cx_pix - cx_full) / f
    yc = (cy_pix - cy_full) / f
    d_cam = np.array([xc, yc, 1.0]); d_cam /= np.linalg.norm(d_cam)
    d_world = R.T @ d_cam
    cam_center = -R.T @ t
    return d_world, cam_center


def _flip_v(m):
    """Flip mask vertically: PIL row 0 (top) → nvdiffrast row 0 (bottom)."""
    return np.ascontiguousarray(m[::-1])


def _safe_crop(m, y0, x0, size, pad_value=0):
    """Crop m[y0:y0+size, x0:x0+size], padding if crop exceeds array bounds.

    Default pad_value=0 (background for uint8 masks).
    """
    h, w = m.shape[:2]
    y1, x1 = min(y0 + size, h), min(x0 + size, w)
    crop = m[y0:y1, x0:x1]
    if crop.shape[0] < size or crop.shape[1] < size:
        full = np.full((size, size), pad_value, dtype=m.dtype)
        full[:crop.shape[0], :crop.shape[1]] = crop
        return full
    return crop


# ─────────────────────────────────────────────────────────────────────────────
# RealScene class
# ─────────────────────────────────────────────────────────────────────────────

class RealScene:
    """Encapsulates a real captured object for the golden topo-carving chain."""

    def __init__(self, scene_dir, device=DEVICE, n_train=64, n_hold=16,
                 res=None, hull_hires=512):
        import pycolmap
        if res is None:
            res = TRAIN_RES
        self.device = device
        self.res     = res
        self.hull_hires = hull_hires
        t_start = time.time()

        # ── Load COLMAP ──────────────────────────────────────────────────────
        rec = pycolmap.Reconstruction(os.path.join(scene_dir, "sparse/0"))
        cam = list(rec.cameras.values())[0]
        f, cx_full, cy_full, _k = cam.params   # SIMPLE_RADIAL; k ignored (|k|<0.02)
        W_orig, H_orig = int(cam.width), int(cam.height)  # 1080, 1920

        all_imgs = sorted(rec.images.values(), key=lambda im: im.name)
        N_reg    = len(all_imgs)   # 145

        # ── Held-out / train split ───────────────────────────────────────────
        hold_idx  = list(range(0, N_reg, max(1, N_reg // n_hold)))[:n_hold]
        hold_set  = set(hold_idx)
        rem_idx   = [i for i in range(N_reg) if i not in hold_set]
        # Uniformly sample n_train from remainder
        train_idx_raw = [rem_idx[int(round(i))] for i in
                         np.linspace(0, len(rem_idx) - 1, n_train)]
        seen = set(); train_idx = []
        for i in train_idx_raw:
            if i not in seen: seen.add(i); train_idx.append(i)
        train_idx = train_idx[:n_train]

        names_train = [all_imgs[i].name for i in train_idx]
        names_hold  = [all_imgs[i].name for i in hold_idx]
        self.names_train = names_train
        self.names_hold  = names_hold
        print(f"[real_scene] {N_reg} registered → {len(names_hold)} held-out, {len(names_train)} train",
              flush=True)

        # ── Load masks (PIL convention: row 0 = top) ─────────────────────────
        mask_dir = os.path.join(scene_dir, "masks_sam2")
        def _load_mask_pil(name):
            stem = os.path.splitext(name)[0]
            return np.array(Image.open(os.path.join(mask_dir, stem + ".png")))

        all_masks_pil = [_load_mask_pil(im.name) for im in all_imgs]

        # ── Compute mask bboxes → crop size S_px ────────────────────────────
        def _bbox(m):
            ys, xs = np.where(m > 127)
            if len(ys) == 0: return None
            return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())

        all_bboxes = [_bbox(m) for m in all_masks_pil]
        valid_bboxes = [b for b in all_bboxes if b is not None]
        max_dim = int(np.percentile([max(b[2]-b[0], b[3]-b[1]) for b in valid_bboxes], 95))   # robust to mask leaks
        S_px = int(math.ceil(1.25 * max_dim))
        print(f"[real_scene] mask bbox max_dim={max_dim}px → S_px={S_px}", flush=True)
        self.S_px = S_px

        def _crop_xy0(bbox):
            if bbox is None: return 0, 0
            cx_obj = (bbox[0] + bbox[2]) / 2
            cy_obj = (bbox[1] + bbox[3]) / 2
            x0 = int(round(cx_obj - S_px / 2))
            y0 = int(round(cy_obj - S_px / 2))
            x0 = max(0, min(x0, W_orig - S_px))
            y0 = max(0, min(y0, H_orig - S_px))
            return x0, y0

        all_xy0 = [_crop_xy0(b) for b in all_bboxes]

        # ── World centre via ray intersection ─────────────────────────────────
        dirs, origs = [], []
        for im, m in zip(all_imgs, all_masks_pil):
            cfw = im.cam_from_world()
            R_i = cfw.rotation.matrix().astype(np.float64)
            t_i = cfw.translation.astype(np.float64)
            d, o = _mask_centroid_ray(m, R_i, t_i, f, cx_full, cy_full)
            if d is not None:
                dirs.append(d); origs.append(o)
        X = _ray_intersection(dirs, origs).astype(np.float64)
        print(f"[real_scene] world centre X={np.round(X, 3)}", flush=True)
        self.X = X

        # ── Scale from mask geometry (no hull needed): object radius per view =
        #    camera distance * (half of the mask bbox max-dim in px) / f ; median over views.
        cam_centers = []
        for im in all_imgs:
            cfw = im.cam_from_world(); R_i = cfw.rotation.matrix().astype(np.float64); t_i = cfw.translation.astype(np.float64)
            cam_centers.append(-R_i.T @ t_i)
        cam_centers = np.stack(cam_centers)
        dists_from_X = np.linalg.norm(cam_centers - X, axis=1)
        d_med = float(np.median(dists_from_X))
        r_views = [dists_from_X[i] * 0.5 * max(b[2]-b[0], b[3]-b[1]) / f for i, b in enumerate(all_bboxes) if b is not None]
        r_obj = float(np.median(r_views))
        s = r_obj / 0.8          # world units per normalised unit: V_norm = (V_world - X) / s puts the object at radius 0.8
        print(f"[real_scene] median camera dist from X: {d_med:.4f} | object radius (median over views) {r_obj:.4f} -> s={s:.4f}", flush=True)
        self.s = s
        # normalised hull bbox: object radius 0.8 -> grid [-1.05, 1.05]^3 (inside every 1.25x crop frustum)
        self._lo_norm = np.full(3, -1.05, np.float32); self._hi_norm = np.full(3, 1.05, np.float32)
        self._lo_pre = X + self._lo_norm * (1.0 / s); self._hi_pre = X + self._hi_norm * (1.0 / s)   # kept for callers that expect them
        hull_pre = None

        def _mvp_for(idx, flip_x_try):
            im = all_imgs[idx]; cfw = im.cam_from_world(); R_i = cfw.rotation.matrix().astype(np.float32); t_i = cfw.translation.astype(np.float32)
            x0i, y0i = all_xy0[idx]; fp = f * res / S_px; cxp = (cx_full - x0i) * res / S_px; cyp = (cy_full - y0i) * res / S_px
            cam_c = -R_i.T @ t_i; dist_n = max(0.01, float(np.linalg.norm((cam_c - X) / s)))
            P_i = _proj_matrix(fp, cxp, cyp, res, max(0.01, 0.1 * dist_n), 15.0 * dist_n, flip_x=flip_x_try)
            return P_i @ _view_matrix_norm(R_i, t_i, X.astype(np.float32), float(s))

        def _hull_norm(idxs, flip_x_try, nres_h=128, hires_h=256):
            """Mask-carved hull in the normalised frame over [-1.05,1.05]^3 (largest component, radius<=1.0)."""
            from hull_field import carve_hull as _carve_hull
            fgs, mv = [], []
            for idx in idxs:
                x0i, y0i = all_xy0[idx]; m_crop = _safe_crop(all_masks_pil[idx], y0i, x0i, S_px)
                m_r = np.array(Image.fromarray(m_crop).resize((hires_h, hires_h), Image.BILINEAR)) > 127
                fgs.append(torch.from_numpy(_flip_v(binary_dilation(m_r, iterations=1))).to(device))
                mv.append(torch.tensor(_mvp_for(idx, flip_x_try), dtype=torch.float32, device=device))
            with torch.no_grad():
                h = _carve_hull(fgs, mv, self._lo_norm, self._hi_norm, nres_h, hires_h, 2, device)
            g = np.linspace(-1.05, 1.05, nres_h); G = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1)
            h &= (np.linalg.norm(G, axis=-1) <= 1.0)
            lab, n_cc = cc_label(h)
            if n_cc > 1: h = lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
            return binary_fill_holes(h)

        # ── Empirical flip probe (flip_x only; y handled by mask flip) ───────
        best_flip_x = False; best_iou = -1.0
        try:
            import nvdiffrast.torch as dr
            from skimage.measure import marching_cubes
            ctx_probe = dr.RasterizeCudaContext()
            test_idxs = train_idx[::8]
            for flip_x_try in (False, True):
                hull_try = _hull_norm(train_idx, flip_x_try)
                spacing = tuple((self._hi_norm - self._lo_norm) / (128 - 1))
                verts_mc, faces_mc, _, _ = marching_cubes(hull_try.astype(np.float32), level=0.5, spacing=spacing)
                verts_mc = (verts_mc + self._lo_norm).astype(np.float32)
                vmc_t = torch.tensor(verts_mc, dtype=torch.float32, device=device)
                fmc_t = torch.tensor(faces_mc.astype(np.int32), dtype=torch.int32, device=device)
                ones_attr = torch.ones(1, len(verts_mc), 3, dtype=torch.float32, device=device)
                ious = []
                for idx in test_idxs:
                    im   = all_imgs[idx]
                    cfw  = im.cam_from_world()
                    R_i  = cfw.rotation.matrix().astype(np.float32)
                    t_i  = cfw.translation.astype(np.float32)
                    x0i, y0i = all_xy0[idx]
                    fp   = f * res / S_px
                    cxp  = (cx_full - x0i) * res / S_px
                    cyp  = (cy_full - y0i) * res / S_px
                    cam_c = -R_i.T @ t_i
                    dist_n = max(0.01, float(np.linalg.norm((cam_c - X) / s)))
                    near_n, far_n = 0.1 * dist_n, 10.0 * dist_n
                    P_try  = _proj_matrix(fp, cxp, cyp, res, near_n, far_n, flip_x=flip_x_try)
                    view_n = _view_matrix_norm(R_i, t_i, X.astype(np.float32), float(s))
                    mvp_try = torch.tensor(P_try @ view_n, dtype=torch.float32, device=device)
                    # Render hull mesh
                    ones_V  = torch.ones(len(verts_mc), 1, dtype=torch.float32, device=device)
                    vh      = torch.cat([vmc_t, ones_V], -1)
                    pos_clip = (mvp_try @ vh.T).T.unsqueeze(0).contiguous()
                    with torch.no_grad():
                        rast, _ = dr.rasterize(ctx_probe, pos_clip, fmc_t, resolution=[res, res])
                        col, _  = dr.interpolate(ones_attr, rast, fmc_t)
                        pos_c2  = pos_clip.contiguous()
                        sil     = dr.antialias(col.contiguous(), rast, pos_c2, fmc_t)[0, :, :, 0]
                    pred_fg = sil.cpu().numpy() > 0.5  # nvdiffrast: row 0 = bottom
                    # GT mask in nvdiffrast convention (flipped)
                    m_crop = _safe_crop(all_masks_pil[idx], y0i, x0i, S_px)
                    m_rsz  = np.array(Image.fromarray(m_crop).resize((res, res), Image.BILINEAR)) > 127
                    gt_fg  = _flip_v(m_rsz)           # row 0 = bottom
                    inter  = float((pred_fg & gt_fg).sum())
                    un     = float((pred_fg | gt_fg).sum())
                    ious.append(inter / max(un, 1))
                iou_med = float(np.median(ious))
                print(f"[real_scene] flip_x={flip_x_try}: IoU={iou_med:.3f}", flush=True)
                if iou_med > best_iou:
                    best_iou = iou_med; best_flip_x = flip_x_try; hull_pre = hull_try; vmc_best, fmc_best = vmc2 = verts_mc, faces_mc
            del ctx_probe
        except Exception as e:
            print(f"[real_scene] flip probe failed ({e}), using flip_x=False", flush=True)

        self.flip_x = best_flip_x
        print(f"[real_scene] chosen flip_x={best_flip_x} IoU(hull vs mask)={best_iou:.3f}", flush=True)

        # ── Build normalised MVPs, views, gt, gtd for training + held-out ─────
        def _build_mvp_view(im, idx):
            cfw  = im.cam_from_world()
            R_i  = cfw.rotation.matrix().astype(np.float32)
            t_i  = cfw.translation.astype(np.float32)
            x0i, y0i = all_xy0[idx]
            fp   = f * res / S_px
            cxp  = (cx_full - x0i) * res / S_px
            cyp  = (cy_full - y0i) * res / S_px
            cam_c = -R_i.T @ t_i
            dist_n = max(0.01, float(np.linalg.norm((cam_c - X) / s)))
            near_n, far_n = max(0.01, 0.1 * dist_n), 15.0 * dist_n
            P_i    = _proj_matrix(fp, cxp, cyp, res, near_n, far_n, flip_x=best_flip_x)
            view_i = _view_matrix_norm(R_i, t_i, X.astype(np.float32), float(s))
            mvp_t  = torch.tensor(P_i @ view_i, dtype=torch.float32, device=device)
            view_t = torch.tensor(view_i,        dtype=torch.float32, device=device)
            proj_t = torch.tensor(P_i,           dtype=torch.float32, device=device)
            return mvp_t, view_t, all_xy0[idx], proj_t

        def _mask_to_gt_nv(m_pil, x0i, y0i):
            """Crop, resize to res, flip vertically (nvdiffrast convention), 0=fg 255=bg."""
            m_crop = _safe_crop(m_pil, y0i, x0i, S_px)
            m_rsz  = np.array(Image.fromarray(m_crop).resize((res, res), Image.BILINEAR)) > 127
            m_nv   = _flip_v(m_rsz)          # row 0 = bottom
            return np.where(m_nv, np.uint8(0), np.uint8(255))  # 0=fg, 255=bg

        # OUT-OF-FRAME pixels: the crop (S_px = 1360) is wider than the portrait frame (1080), and the object can leave
        # the frame (tail/head). _safe_crop pads with 0 = background, which CARVED the object there. Treat padding as
        # unknown: excluded from the silhouette loss (self.valid) and never carved (hull masks set to foreground there).
        valid_full = np.full((H_orig, W_orig), 255, np.uint8)
        def _valid_nv(x0i, y0i, r):
            v = _safe_crop(valid_full, y0i, x0i, S_px); v = np.array(Image.fromarray(v).resize((r, r), Image.NEAREST)) > 127
            return _flip_v(v)
        def _mask_to_hires_nv(m_pil, x0i, y0i, hires):
            m_crop = _safe_crop(m_pil, y0i, x0i, S_px)
            m_rsz  = np.array(Image.fromarray(m_crop).resize((hires, hires), Image.BILINEAR)) > 127
            m_rsz  = binary_dilation(m_rsz, iterations=1)  # match dilate=True
            m_rsz |= ~_flip_v(_valid_nv(x0i, y0i, hires))   # out of frame -> "foreground" -> not carved (m_rsz is still in PIL row order here)
            m_nv   = _flip_v(m_rsz)
            return torch.from_numpy(m_nv)

        # ── Photo / mono-depth loaders (same crop/resize/flip as masks) ─────
        image_dir = os.path.join(scene_dir, "images")
        depth_dir = os.path.join(scene_dir, "depth_da2")
        _has_depth = os.path.isdir(depth_dir)

        def _load_mono_nv(name, x0i, y0i):
            """Load cached DA2 depth map, crop/resize/flip exactly like masks.
            Returns float32 [res,res] inverse-depth (larger=nearer). 0.0 where missing."""
            if not _has_depth:
                return np.zeros((res, res), dtype=np.float32)
            stem = os.path.splitext(name)[0]
            npy_path = os.path.join(depth_dir, stem + ".npy")
            try:
                mono = np.load(npy_path).astype(np.float32)   # [H_full, W_full]
                h_img, w_img = mono.shape[:2]
                full = np.zeros((S_px, S_px), dtype=np.float32)   # pad with 0
                y1 = min(y0i + S_px, h_img); x1 = min(x0i + S_px, w_img)
                full[:y1 - y0i, :x1 - x0i] = mono[y0i:y1, x0i:x1]
                m_rsz = np.array(Image.fromarray(full).resize((res, res), Image.BILINEAR))
                return _flip_v(m_rsz)                          # float32 [res,res]
            except Exception:
                return np.zeros((res, res), dtype=np.float32)

        def _load_photo_nv(name, x0i, y0i):
            """Load photo, crop/resize exactly like masks → float32 [res,res,3] in [0,1]."""
            stem = os.path.splitext(name)[0]
            img_path = os.path.join(image_dir, stem + ".jpg")
            try:
                img = np.array(Image.open(img_path).convert("RGB"))
                h_img, w_img = img.shape[:2]
                full = np.zeros((S_px, S_px, 3), dtype=np.uint8)
                y1 = min(y0i + S_px, h_img); x1 = min(x0i + S_px, w_img)
                full[:y1-y0i, :x1-x0i] = img[y0i:y1, x0i:x1]
                m_rsz = np.array(Image.fromarray(full).resize((res, res), Image.BILINEAR))
                return _flip_v(m_rsz).astype(np.float32) / 255.0
            except Exception:
                return np.full((res, res, 3), 0.5, np.float32)

        # Training
        train_mvps_list, train_views_list, train_gt_list = [], [], []
        train_masks_hires_list = []; train_valid_list = []
        train_proj_list = []; train_rgb_list = []; train_mono_list = []
        for idx in train_idx:
            im  = all_imgs[idx]
            mvp_i, view_i, (x0i, y0i), proj_i = _build_mvp_view(im, idx)
            train_mvps_list.append(mvp_i)
            train_views_list.append(view_i)
            train_proj_list.append(proj_i)
            train_gt_list.append(_mask_to_gt_nv(all_masks_pil[idx], x0i, y0i))
            train_valid_list.append(_valid_nv(x0i, y0i, res))
            train_masks_hires_list.append(_mask_to_hires_nv(all_masks_pil[idx], x0i, y0i, hull_hires))
            train_rgb_list.append(_load_photo_nv(im.name, x0i, y0i))
            train_mono_list.append(_load_mono_nv(im.name, x0i, y0i))

        # Held-out
        hold_mvps_list, hold_gt_list = [], []
        hold_views_list = []
        hold_rgb_list = []; hold_mono_list = []
        for idx in hold_idx:
            im  = all_imgs[idx]
            mvp_i, view_i, (x0i, y0i), _ = _build_mvp_view(im, idx)
            hold_mvps_list.append(mvp_i)
            hold_views_list.append(view_i)
            hold_gt_list.append(_mask_to_gt_nv(all_masks_pil[idx], x0i, y0i))
            hold_rgb_list.append(_load_photo_nv(im.name, x0i, y0i))
            hold_mono_list.append(_load_mono_nv(im.name, x0i, y0i))

        self.mvps        = torch.stack(train_mvps_list)      # [64,4,4]
        self.views       = torch.stack(train_views_list)     # [64,4,4]
        self.proj        = torch.stack(train_proj_list)      # [64,4,4] projection matrices
        self.view_w2c    = self.views                        # alias: view_w2c == views (world→cam, OpenGL, normalised)
        self.gt          = np.stack(train_gt_list)           # [64,res,res] uint8, 0=fg
        self.valid       = np.stack(train_valid_list)        # [64,res,res] bool: inside the original frame (loss weight)
        self.gtd         = np.zeros((len(train_idx), res, res), np.float32)
        self.gtdiff      = np.zeros((len(train_idx), res, res), np.float32)
        self.max_r       = 0.8
        self.masks_hires = train_masks_hires_list
        self.rgb         = np.stack(train_rgb_list)          # [64,res,res,3] float32 [0,1]
        self.mono        = np.stack(train_mono_list)         # [64,res,res] float32 inv-depth (0 if no cache)
        self.ho_mvps     = torch.stack(hold_mvps_list)        # [16,4,4]
        self.ho_view_w2c = torch.stack(hold_views_list)      # [16,4,4] world→cam for held-out
        self.ho_gt       = np.stack(hold_gt_list)            # [16,res,res] uint8
        self.ho_rgb      = np.stack(hold_rgb_list)           # [16,res,res,3] float32 [0,1]
        self.ho_mono     = np.stack(hold_mono_list)          # [16,res,res] float32 inv-depth
        self._has_depth  = _has_depth
        self._hull_hf    = None
        self._hull_pre   = hull_pre
        # Sanity: proj @ view_w2c == mvps (float32 rounding may give ~1e-6)
        with torch.no_grad():
            _mvp_check = torch.bmm(self.proj, self.view_w2c)
            _diff = (_mvp_check - self.mvps).abs().max().item()
            print(f"[real_scene] proj@view_w2c sanity: max_abs_diff={_diff:.2e}", flush=True)

        # ── Final IoU validation on all 64 training views ────────────────────
        try:
            import nvdiffrast.torch as dr
            from skimage.measure import marching_cubes
            from scipy import ndimage as ndi
            vmc2, fmc2 = vmc_best, fmc_best
            vmc_t = torch.tensor(vmc2, dtype=torch.float32, device=device)
            fmc_t = torch.tensor(fmc2.astype(np.int32), dtype=torch.int32, device=device)
            ctx_v = dr.RasterizeCudaContext()
            ones_v = torch.ones(1, len(vmc2), 3, dtype=torch.float32, device=device)
            ious_val = []
            for k in range(len(train_mvps_list)):
                mvp_k = train_mvps_list[k]
                ones_vk = torch.ones(len(vmc2), 1, dtype=torch.float32, device=device)
                vh_k = torch.cat([vmc_t, ones_vk], -1)
                pos_k = (mvp_k @ vh_k.T).T.unsqueeze(0).contiguous()
                with torch.no_grad():
                    rast_k, _ = dr.rasterize(ctx_v, pos_k, fmc_t, resolution=[res, res])
                    col_k, _  = dr.interpolate(ones_v, rast_k, fmc_t)
                    sil_k     = dr.antialias(col_k.contiguous(), rast_k, pos_k.contiguous(), fmc_t)[0, :, :, 0]
                pred_fg_k = sil_k.cpu().numpy() > 0.5
                gt_fg_k   = train_gt_list[k] < 128  # 0=fg → True=fg ✓
                inter_k = float((pred_fg_k & gt_fg_k).sum())
                un_k    = float((pred_fg_k | gt_fg_k).sum())
                ious_val.append(inter_k / max(un_k, 1))
            del ctx_v
            iou_med_val = float(np.median(ious_val))
            iou_min_val = float(np.min(ious_val))
            print(f"[real_scene] IoU(hull_render vs mask): median={iou_med_val:.3f} min={iou_min_val:.3f}",
                  flush=True)
            if iou_med_val < 0.85:
                print("[real_scene] WARNING: median IoU < 0.85 – check camera conventions", flush=True)
        except Exception as e:
            print(f"[real_scene] IoU validation skipped ({e})", flush=True)
            iou_med_val = iou_min_val = float("nan")

        elapsed = time.time() - t_start
        print(f"[real_scene] loaded {elapsed:.1f}s | S_px={S_px} flip_x={best_flip_x} "
              f"train={len(names_train)} hold={len(names_hold)}", flush=True)

    # ── Public methods ────────────────────────────────────────────────────────

    def hull(self, ctx, extra_pts=None, nres=256, hires=512, vote=None):
        if vote is None: vote = int(os.environ.get("HULL_VOTE", "2"))   # real data: views disagree by a few px -> allow more dissenting views before carving
        """Build (or return cached) HullField from training mask crops."""
        if self._hull_hf is not None:
            return self._hull_hf
        from hull_field import carve_hull as _carve_hull, HullField

        if hires != self.hull_hires:
            fgs_h = []
            for m in self.masks_hires:
                m_np  = m.cpu().numpy().astype(np.uint8) * 255
                m_r   = np.array(Image.fromarray(m_np).resize((hires, hires), Image.NEAREST)) > 127
                m_r   = binary_dilation(m_r, iterations=1)
                # Already in nvdiffrast convention (was flipped during creation)
                fgs_h.append(torch.from_numpy(m_r))
        else:
            fgs_h = self.masks_hires

        fgs_h = [f.to(self.device) for f in fgs_h]

        lo_norm = self._lo_norm.copy(); hi_norm = self._hi_norm.copy()
        if extra_pts is not None:
            ep = np.asarray(extra_pts)
            lo_norm = np.minimum(lo_norm, ep.min(0) - 0.02)
            hi_norm = np.maximum(hi_norm, ep.max(0) + 0.02)

        with torch.no_grad():
            hull_arr = _carve_hull(fgs_h, self.mvps, lo_norm, hi_norm, nres, hires, vote, self.device)

        lab, n_cc = cc_label(hull_arr)          # real data: permissive voting leaves floaters -> keep the largest component
        if n_cc > 1: hull_arr = lab == (1 + np.argmax(np.bincount(lab.ravel())[1:]))
        hf = HullField(lo_norm, hi_norm, hull_arr, self.device)
        hf.hull = hull_arr
        self._hull_hf = hf
        print(f"[real_scene] hull: {hull_arr.sum()} voxels nres={nres} vote={vote}", flush=True)
        return hf

    def heldout_exam(self, ctx, v, t):
        """Same signature as run_64v.heldout_exam but on held-out real masks."""
        import nvdiffrast.torch as dr
        pvt = torch.tensor(np.asarray(v, np.float32), dtype=torch.float32, device=self.device)
        pft = torch.tensor(np.asarray(t, np.int32),   dtype=torch.int32,   device=self.device)
        ones = torch.ones(pvt.shape[0], 1, device=self.device); vh = torch.cat([pvt, ones], -1); ps = []
        with torch.no_grad():
            for mvp in self.ho_mvps:                                   # render at the scene resolution (ho_gt is res x res)
                pos = (mvp @ vh.T).T.unsqueeze(0).contiguous()
                rast, _ = dr.rasterize(ctx, pos, pft, resolution=[self.res, self.res]); ps.append((rast[0, :, :, 3] > 0).float().cpu().numpy())
        ps = np.stack(ps)
        inter = un = hair = 0; maxblob = 0
        for i in range(len(self.ho_mvps)):
            g = self.ho_gt[i] < 128   # fg (flipped, nvdiffrast convention)
            p = ps[i] > 0.5           # rendered fg (nvdiffrast convention)
            inter += int((g & p).sum()); un += int((g | p).sum())
            out = p & ~binary_dilation(g, iterations=2)
            hair += int(out.sum())
            lab, nb = cc_label(out)
            for k in range(1, nb + 1):
                maxblob = max(maxblob, int((lab == k).sum()))
        return inter / max(un, 1), hair, maxblob


def load_real_scene(scene_dir, device=DEVICE, n_train=64, n_hold=16,
                    res=None, hull_hires=512):
    return RealScene(scene_dir, device=device, n_train=n_train, n_hold=n_hold,
                     res=res, hull_hires=hull_hires)


# ─────────────────────────────────────────────────────────────────────────────
# Self-test / acceptance test 1
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 despike/real_scene.py <scene_dir>"); sys.exit(1)
    scene_dir = sys.argv[1]
    scene = load_real_scene(scene_dir)
    print(f"\n--- acceptance test 1 ---")
    print(f"train/hold split: {len(scene.names_train)}/{len(scene.names_hold)}")
    print(f"S_px={scene.S_px}  flip_x={scene.flip_x}")
    print(f"s={scene.s:.4f}  max_r={scene.max_r}")
    from hull_field import hull_genus as _hull_genus
    g_pre, gr = _hull_genus(scene._hull_pre)
    print(f"normalised hull (128^3) voxels={int(scene._hull_pre.sum())}  genus={g_pre} per_r={gr}")
