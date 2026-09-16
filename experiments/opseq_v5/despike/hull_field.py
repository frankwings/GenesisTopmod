"""Voting visual-hull evidence field (shared by phase1f/1g).

Space-carving evidence (Boss, 2026-09-01): a pixel outside the GT silhouette
proves no material on that ray. Voting version (>=VOTE views required) is
robust to subpixel-thin structures vanishing from single rasterized views
(the old strict intersection ate 3.05% of GT incl. the hand).
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import distance_transform_edt
import nvdiffrast.torch as dr


class HullField:
    """EDT distance-to-hull field, trilinear-sampled, world units."""

    def __init__(self, lo, hi, hull, device):
        self.lo, self.hi = lo, hi
        self.nres = hull.shape[0]
        self.sp = (hi - lo) / (self.nres - 1)
        self.pitch = float(np.max(self.sp))
        dist = distance_transform_edt(~hull, sampling=tuple(self.sp))
        self.vol = (torch.from_numpy(dist.astype(np.float32))
                    .unsqueeze(0).unsqueeze(0).to(device))
        # Phase 1h: inside distance-to-boundary (hull boundary ~ GT surface
        # at 64v, volume ratio 0.996). Surface points deep inside hull =
        # missing material above them (unfilled finger tubes).
        din = distance_transform_edt(hull, sampling=tuple(self.sp))
        self.vol_in = (torch.from_numpy(din.astype(np.float32))
                       .unsqueeze(0).unsqueeze(0).to(device))
        self.lo_t = torch.tensor(lo, dtype=torch.float32, device=device)
        self.hi_t = torch.tensor(hi, dtype=torch.float32, device=device)

    def _sample(self, vol, pts_t):
        g = 2.0 * (pts_t - self.lo_t) / (self.hi_t - self.lo_t) - 1.0
        grid = g[:, [2, 1, 0]].view(1, 1, 1, -1, 3)   # vol dims (X,Y,Z)
        d = F.grid_sample(vol, grid, mode="bilinear",
                          padding_mode="border", align_corners=True)
        return d.view(-1)

    def dist(self, pts_t):
        """[N,3] world points -> [N] distance outside hull (0 inside)."""
        return self._sample(self.vol, pts_t)

    def dist_in(self, pts_t):
        """[N,3] -> [N] distance to hull boundary from inside (0 outside)."""
        return self._sample(self.vol_in, pts_t)


def carve_hull(fgs, mvps, lo, hi, nres, hires, vote, device):
    """Space-carving voxel loop shared by synthetic and real paths.

    fgs   : list of N bool tensors [hires, hires] – True = foreground (inside object).
    mvps  : list/tensor of N [4,4] MVP matrices that project world → clip for the hires image.
    lo/hi : bbox of the voxel grid (length-3 arrays, world units).
    Returns hull bool array [nres, nres, nres] (True = inside object)."""
    axes = [torch.linspace(float(lo[a]), float(hi[a]), nres, device=device)
            for a in range(3)]
    gx, gy, gz = torch.meshgrid(*axes, indexing="ij")
    P = torch.stack([gx, gy, gz], -1).view(-1, 3)
    Ph = torch.cat([P, torch.ones(P.shape[0], 1, device=device)], 1)
    votes = torch.zeros(P.shape[0], dtype=torch.uint8, device=device)
    mvps_list = mvps if isinstance(mvps, (list, tuple)) else [mvps[i] for i in range(len(mvps))]
    for i, mvp_i in enumerate(mvps_list):
        mvp_t = mvp_i if isinstance(mvp_i, torch.Tensor) else torch.tensor(mvp_i, dtype=torch.float32, device=device)
        clip = (mvp_t.to(device) @ Ph.T).T
        w = clip[:, 3].clamp(min=1e-8)
        x_ndc, y_ndc = clip[:, 0] / w, clip[:, 1] / w
        ui = ((x_ndc + 1) * 0.5 * hires).long().clamp(0, hires - 1)
        vi = ((y_ndc + 1) * 0.5 * hires).long().clamp(0, hires - 1)
        inb = (x_ndc.abs() <= 1) & (y_ndc.abs() <= 1)
        fg_i = fgs[i].to(device) if not fgs[i].is_cuda else fgs[i]
        votes += (inb & ~fg_i[vi, ui]).to(torch.uint8)
    return (votes < vote).view(nres, nres, nres).cpu().numpy()


def build_vote_hull(ctx, mvps, gv_norm, gf, extra_pts, device,
                    nres=256, hires=512, vote=2, dilate=True, ss_thr=None):
    """dilate: 1px dilation of inside (safe but fat, hull/GT~1.13).
    ss_thr: if set, 2x supersampled coverage >= ss_thr defines inside
    (unbiased edge; hires=1024, ss_thr=0.25, vote=2 -> hull/GT 1.059,
    GT erosion 0.00%). Phase 1i tight mode."""
    """Hull from GT training silhouettes. gv_norm: normalized GT verts.
    extra_pts: e.g. current mesh verts, only to widen the voxel bbox."""
    from pipeline.cameras import transform_to_clip
    gvt = torch.tensor(gv_norm, dtype=torch.float32, device=device)
    gft = torch.tensor(np.asarray(gf, np.int32), dtype=torch.int32,
                       device=device)
    lo = gv_norm.min(0); hi = gv_norm.max(0)
    if extra_pts is not None:
        lo = np.minimum(lo, np.asarray(extra_pts).min(0))
        hi = np.maximum(hi, np.asarray(extra_pts).max(0))
    lo, hi = lo - 0.02, hi + 0.02
    fgs = []
    with torch.no_grad():
        for i in range(len(mvps)):
            pos = transform_to_clip(gvt, mvps[i])
            if ss_thr is not None:
                rast, _ = dr.rasterize(ctx, pos, gft,
                                       resolution=[2 * hires, 2 * hires])
                cov = F.avg_pool2d((rast[0, :, :, 3] > 0).float()[None, None], 2)
                fgs.append(cov[0, 0] >= ss_thr)
                continue
            rast, _ = dr.rasterize(ctx, pos, gft, resolution=[hires, hires])
            fg = (rast[0, :, :, 3] > 0).float()
            if dilate:
                fg = (F.max_pool2d(fg[None, None], 3, 1, 1)[0, 0] > 0)
            else:
                fg = fg > 0
            fgs.append(fg)
        hull = carve_hull(fgs, mvps, lo, hi, nres, hires, vote, device)
    hf = HullField(lo, hi, hull, device)
    hf.hull = hull
    return hf


def hull_genus(hull, radii=(1, 2, 3)):
    """Genus of the space-carved hull (bool voxel grid, True = inside) after morphological
    closing+opening of r voxels, largest component, cavities filled. Solid Euler number chi = 1 - g.
    Returns (persistent genus = mode over radii, {r: g}). LESSONS 24: equals GT genus on all 5 shapes."""
    import numpy as np
    from scipy import ndimage as ndi
    from skimage.measure import euler_number
    occ = np.asarray(hull).astype(bool); S3 = ndi.generate_binary_structure(3, 1); out = {}
    for r in radii:
        v = ndi.binary_opening(ndi.binary_closing(occ, S3, iterations=r), S3, iterations=r)
        lab, n = ndi.label(v)
        if n == 0: out[r] = 0; continue
        big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
        out[r] = int(1 - euler_number(ndi.binary_fill_holes(lab == big), connectivity=1))
    vals = list(out.values()); g = max(set(vals), key=lambda x: (vals.count(x), -x))
    return g, out
