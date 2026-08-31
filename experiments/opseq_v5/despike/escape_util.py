"""B-approach: project vertices to the 6 TRAINING views and flag those that
land OUTSIDE the GT silhouette (they create area the target does not have).
Only the 6 training views are used -> obeys the 'no multi-view burn' rule.
"""
import sys
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import numpy as np, torch
from scipy.ndimage import binary_dilation
from pipeline.cameras import transform_to_clip


@torch.no_grad()
def vert_pixels(verts, mvp, H, W):
    """Return integer (row, col) pixel of each vertex under mvp, matching the
    nvdiffrast rasterizer convention (row 0 = NDC y = +1 top)."""
    pc = transform_to_clip(verts, mvp)[0]          # [V,4]
    w = pc[:, 3:4].clamp(min=1e-6)
    ndc = pc[:, :3] / w                            # x,y in [-1,1]
    col = ((ndc[:, 0] * 0.5 + 0.5) * (W - 1)).round().long().clamp(0, W - 1)
    row = ((ndc[:, 1] * 0.5 + 0.5) * (H - 1)).round().long().clamp(0, H - 1)
    return row, col


@torch.no_grad()
def escape_mask(verts, mvps, gt_uint8, dilate=2):
    """verts[V,3] -> bool[V]: True if the vertex projects OUTSIDE dilate(GT_fg)
    in ANY of the given views. gt_uint8[N,H,W] (<128 = fg)."""
    V = verts.shape[0]; N, H, W = gt_uint8.shape
    dev = verts.device
    out = torch.zeros(V, dtype=torch.bool, device=dev)
    for i in range(N):
        gt_fg = gt_uint8[i] < 128
        gt_in = binary_dilation(gt_fg, iterations=dilate)
        gt_in_t = torch.from_numpy(gt_in).to(dev)
        row, col = vert_pixels(verts, mvps[i], H, W)
        inside = gt_in_t[row, col]
        out |= ~inside
    return out
