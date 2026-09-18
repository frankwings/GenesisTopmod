"""
Batched 64-view losses for the DR loop.

These produce the same scalar as the per-view loop without any GPU syncs:
  - No .item() calls
  - No per-view mask branch (replaced by torch.where)

Usage:
    sil_b, ndc_b, fg_b, diff_b = render_sdd_batch(ctx, verts_t, faces_t, mvps, views)
    sl = sil_loss_batch(sil_b, targets[..., 0])       # targets [N,H,W,1] -> [N,H,W]
    dl = depth_loss_batch(ndc_b, fg_b, gtd_stack, gtfg_stack)
    fl = diff_loss_batch(diff_b, gtdf_stack)
"""
import torch
import torch.nn.functional as F


def sil_loss_batch(sil_batch, targets_batch):
    """L1 silhouette loss averaged over N*H*W.

    sil_batch     : [N,H,W] float  (from render_sdd_batch)
    targets_batch : [N,H,W] float  (targets[..., 0])

    Identical to sum_i F.l1_loss(sil_i[0], targets_i) / N because
    F.l1_loss means over all elements regardless of batching.
    """
    w = VALID
    if w is None: return F.l1_loss(sil_batch, targets_batch)
    return ((sil_batch - targets_batch).abs() * w).sum() / w.sum().clamp_min(1.0)

VALID = None   # optional [N,H,W] float weight (real data: 0 outside the original frame); set by run_64v / phase4_inloop


def depth_loss_batch(ndc_z_batch, fg_batch, gtd_stack, gtfg_stack):
    """Masked depth L1 loss, no .item() GPU syncs.

    Replicates depth_loss_masked per view:
      mask_i = fg_i & gtfg_i
      if mask_i.sum() < 4: 0   else: |ndc_z_i[mask_i] - gtd_i[mask_i]|.mean()
    Returns sum_i per_view_i / N.

    ndc_z_batch : [N,H,W] float
    fg_batch    : [N,H,W] bool
    gtd_stack   : [N,H,W] float
    gtfg_stack  : [N,H,W] bool
    """
    mask = fg_batch & gtfg_stack                                    # [N,H,W]
    cnt  = mask.sum(dim=(1, 2))                                     # [N]
    err  = ((ndc_z_batch - gtd_stack).abs() * mask.float()).sum(dim=(1, 2))  # [N]
    per_view = torch.where(cnt >= 4, err / cnt.clamp(min=1),
                           torch.zeros_like(err))                   # [N]
    return per_view.sum() / ndc_z_batch.shape[0]


def diff_loss_batch(diff_batch, gtdf_stack):
    """L1 diffuse loss averaged over N*H*W.

    diff_batch  : [N,H,W] float  (from render_sdd_batch)
    gtdf_stack  : [N,H,W] float

    Identical to sum_i F.l1_loss(diff_i, gtdf_i) / N.
    """
    return F.l1_loss(diff_batch, gtdf_stack)


def soft_targets(targets, sigma_px):
    """Gaussian-blur binary silhouette targets [N,H,W,1] (or [N,H,W]) with sigma in pixels.
    Boss 2026-09-17: real views disagree by a few px (pose noise); a soft target lets DR settle on the AVERAGE boundary
    instead of being pulled by whichever view it is closest to (a slightly misplaced thin arm costs little, a missing arm
    costs its full area). sigma_px <= 0 returns the input unchanged (synthetic path)."""
    import torch, torch.nn.functional as F, math
    if sigma_px <= 0: return targets
    squeeze = targets.dim() == 4
    x = targets[..., 0] if squeeze else targets
    r = int(math.ceil(3 * sigma_px)); k = torch.arange(-r, r + 1, device=x.device, dtype=x.dtype)
    g = torch.exp(-0.5 * (k / sigma_px) ** 2); g = g / g.sum()
    y = x.unsqueeze(1)
    y = F.conv2d(F.pad(y, (r, r, 0, 0), mode="replicate"), g.view(1, 1, 1, -1))
    y = F.conv2d(F.pad(y, (0, 0, r, r), mode="replicate"), g.view(1, 1, -1, 1))[:, 0]
    return y.unsqueeze(-1) if squeeze else y
