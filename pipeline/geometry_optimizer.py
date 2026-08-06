"""
Differentiable geometry optimizer using nvdiffrast.

Given a topology mesh (fixed face connectivity, learnable vertex positions)
and target silhouette images from multiple views, optimizes vertex positions
via gradient descent through a differentiable rasterizer.

Public API
----------
render_silhouette(ctx, verts, faces, mvp, resolution) -> [1, H, W, 1]
    Render a silhouette image (differentiable w.r.t. verts).

laplacian_loss(verts, faces)          -> scalar tensor
edge_length_loss(verts, faces)        -> scalar tensor
normal_consistency_loss(verts, faces) -> scalar tensor

optimize(ctx, verts_init, faces, target_images, mvps, **kwargs)
    -> (verts_final, loss_history)
"""

from __future__ import annotations
import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr

from .cameras import transform_to_clip


# ── differentiable silhouette rendering ───────────────────────────────────────

def render_silhouette(
    ctx:        dr.RasterizeCudaContext,
    verts:      torch.Tensor,   # [V, 3]  float32  (differentiable)
    faces:      torch.Tensor,   # [F, 3]  int32
    mvp:        torch.Tensor,   # [4, 4]  float32
    resolution: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    Return a silhouette image [1, H, W, 1] with values in [0, 1].

    The output is differentiable w.r.t. *verts* via nvdiffrast antialias.
    """
    H, W = resolution
    V = verts.shape[0]

    # Transform to clip space
    pos_clip = transform_to_clip(verts, mvp)   # [1, V, 4]  contiguous

    # Rasterize
    rast, _ = dr.rasterize(ctx, pos_clip, faces, resolution=[H, W])

    # Interpolate a constant white attribute (any constant works for silhouette)
    color_attr = torch.ones(1, V, 3, dtype=torch.float32, device=verts.device)
    color, _ = dr.interpolate(color_attr, rast, faces)

    # Antialias: makes silhouette boundary differentiable w.r.t. vertex positions
    color = dr.antialias(color, rast, pos_clip, faces)

    return color[..., :1]   # [1, H, W, 1]  silhouette channel


def render_batch(
    ctx:        dr.RasterizeCudaContext,
    verts:      torch.Tensor,          # [V, 3]
    faces:      torch.Tensor,          # [F, 3]
    mvps:       torch.Tensor,          # [N, 4, 4]
    resolution: Tuple[int, int] = (256, 256),
) -> torch.Tensor:
    """
    Render from N views and return [N, H, W, 1] silhouette batch.
    """
    N = mvps.shape[0]
    results = []
    for i in range(N):
        sil = render_silhouette(ctx, verts, faces, mvps[i], resolution)
        results.append(sil)          # each [1, H, W, 1]
    return torch.cat(results, dim=0)  # [N, H, W, 1]


# ── regularizers ─────────────────────────────────────────────────────────────

def laplacian_loss(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Uniform Laplacian smoothness loss.

    L[i] = mean(v[j] for j in neighbours(i)) - v[i]
    loss  = mean(||L[i]||²)
    """
    V = verts.shape[0]
    device = verts.device

    lap = torch.zeros(V, 3, dtype=verts.dtype, device=device)
    deg = torch.zeros(V, 1, dtype=verts.dtype, device=device)
    ones = torch.ones(faces.shape[0], 1, dtype=verts.dtype, device=device)

    for k in range(3):
        src = faces[:, k].long()
        dst = faces[:, (k + 1) % 3].long()
        lap.index_add_(0, src, verts[dst])
        deg.index_add_(0, src, ones)

    # mean of neighbour positions minus self
    lap = lap / deg.clamp(min=1) - verts
    return (lap ** 2).sum(dim=-1).mean()


def edge_length_loss(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Penalise deviations from the mean edge length (prevents degeneracy).

    loss = mean((||e||² - mean_len²)²)  … variance of squared edge lengths.
    """
    v0 = verts[faces[:, 0].long()]
    v1 = verts[faces[:, 1].long()]
    v2 = verts[faces[:, 2].long()]

    l01 = (v0 - v1).norm(dim=-1)
    l12 = (v1 - v2).norm(dim=-1)
    l20 = (v2 - v0).norm(dim=-1)

    lengths = torch.cat([l01, l12, l20])
    mean_len = lengths.mean().detach()
    return ((lengths - mean_len) ** 2).mean()


def normal_consistency_loss(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Penalise faces whose normals disagree with adjacent faces.

    Encourages smooth curvature variation (prevents kinks).
    """
    v0 = verts[faces[:, 0].long()]
    v1 = verts[faces[:, 1].long()]
    v2 = verts[faces[:, 2].long()]

    # Face normals (unnormalised)
    n = torch.linalg.cross(v1 - v0, v2 - v0)   # [F, 3]
    n_norm = n / (n.norm(dim=-1, keepdim=True) + 1e-8)

    # Build edge → face map: for each directed edge (i→j) record the face normal
    # Adjacency: same undirected edge (i, j) appears twice with opposite orientation
    F_count = faces.shape[0]
    device  = verts.device

    # Collect directed edges and their face normals
    src_dirs: List[torch.Tensor] = []
    normals:  List[torch.Tensor] = []
    for k in range(3):
        src = faces[:, k].long()
        dst = faces[:, (k + 1) % 3].long()
        edge_id = src * verts.shape[0] + dst   # unique directed-edge key
        src_dirs.append(edge_id)
        normals.append(n_norm)

    src_dirs_t  = torch.cat(src_dirs)           # [3F]
    normals_t   = torch.cat(normals, dim=0)     # [3F, 3]

    # Accumulate normals per (i→j) key — adjacent faces share reversed edge (j→i)
    V = verts.shape[0]
    normal_acc = torch.zeros(V * V, 3, dtype=verts.dtype, device=device)
    count_acc  = torch.zeros(V * V, 1, dtype=verts.dtype, device=device)
    ones = torch.ones(3 * F_count, 1, dtype=verts.dtype, device=device)

    normal_acc.index_add_(0, src_dirs_t, normals_t)
    count_acc.index_add_ (0, src_dirs_t, ones)

    # Reversed keys for the neighbouring face
    rev_dirs = torch.cat([
        faces[:, (k + 1) % 3].long() * V + faces[:, k].long()
        for k in range(3)
    ])                                           # [3F]  reversed edges
    twin_normals = normal_acc[rev_dirs]          # [3F, 3]  normals of twin faces
    twin_count   = count_acc [rev_dirs]          # [3F, 1]

    mask = (twin_count.squeeze(-1) > 0)
    if mask.sum() == 0:
        return torch.tensor(0.0, device=device)

    dot = (normals_t[mask] * twin_normals[mask]).sum(dim=-1)
    return (1.0 - dot).mean()


def mesh_volume(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    Signed volume of the mesh via the divergence theorem:
        V = Σ_f det(v0, v1, v2) / 6
    Differentiable w.r.t. verts.  Positive for outward-wound closed meshes.
    """
    v0 = verts[faces[:, 0].long()]
    v1 = verts[faces[:, 1].long()]
    v2 = verts[faces[:, 2].long()]
    return (torch.linalg.cross(v0, v1) * v2).sum() / 6.0


# ── main optimizer ────────────────────────────────────────────────────────────

def optimize(
    ctx:           dr.RasterizeCudaContext,
    verts_init:    torch.Tensor,           # [V, 3] float32
    faces:         torch.Tensor,           # [F, 3] int32
    target_images: torch.Tensor,           # [N, H, W, 1] float32
    mvps:          torch.Tensor,           # [N, 4, 4] float32
    num_steps:     int   = 1000,
    lr:            float = 3e-3,
    lambda_lap:    float = 0.05,
    lambda_edge:   float = 0.01,
    lambda_normal: float = 0.0,
    lambda_vol:    float = 0.0,
    vol_min_ratio: float = 0.25,
    resolution:    Tuple[int, int] = (256, 256),
    log_every:     int   = 50,
    scheduler:     bool  = True,
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Optimise vertex positions to match *target_images* silhouettes.

    Parameters
    ----------
    ctx            : nvdiffrast CUDA rasterize context.
    verts_init     : Initial vertex positions [V, 3].  Copied internally.
    faces          : Triangle face indices [F, 3] int32.  Not modified.
    target_images  : Silhouette targets [N, H, W, 1].
    mvps           : One MVP matrix per view [N, 4, 4].
    num_steps      : Gradient descent iterations.
    lr             : Adam learning rate.
    lambda_lap     : Laplacian smoothness weight.
    lambda_edge    : Edge length regularisation weight.
    lambda_normal  : Normal consistency weight (0 = disabled).
    lambda_vol     : Volume-preservation hinge weight (0 = disabled).
                     Penalises the mesh only when its volume drops below
                     vol_min_ratio × initial volume.  Prevents the classic
                     single-view degeneracy where depth is unconstrained and
                     the mesh collapses into a paper-thin sheet facing the
                     camera (silhouette loss cannot see depth).
    vol_min_ratio  : Fraction of the initial volume below which the hinge
                     activates.
    resolution     : Render resolution (H, W).
    log_every      : Print progress every N steps.
    scheduler      : If True, use cosine LR annealing.

    Returns
    -------
    verts_final : [V, 3] optimised vertex positions (detached).
    history     : list of {step, loss, sil_loss, reg_loss} dicts.
    """
    device = verts_init.device
    N = mvps.shape[0]

    # Learnable vertex positions
    verts = verts_init.clone().detach().to(device).requires_grad_(True)

    # Reference volume for the anti-flattening hinge
    vol_init = mesh_volume(verts_init, faces).abs().detach().clamp(min=1e-8)

    optimizer = torch.optim.Adam([verts], lr=lr)
    if scheduler:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_steps, eta_min=lr * 0.01)

    history: List[Dict] = []

    for step in range(num_steps):
        optimizer.zero_grad()

        # ── Silhouette loss across all views ──────────────────────────
        sil_loss = torch.tensor(0.0, device=device)
        for i in range(N):
            rendered = render_silhouette(ctx, verts, faces, mvps[i], resolution)  # [1,H,W,1]
            target_i  = target_images[i:i+1]                                      # [1,H,W,1]
            sil_loss  = sil_loss + F.l1_loss(rendered, target_i)
        sil_loss = sil_loss / N

        # ── Regularisers ──────────────────────────────────────────────
        reg_loss = torch.tensor(0.0, device=device)
        if lambda_lap   > 0:
            reg_loss = reg_loss + lambda_lap   * laplacian_loss(verts, faces)
        if lambda_edge  > 0:
            reg_loss = reg_loss + lambda_edge  * edge_length_loss(verts, faces)
        if lambda_normal > 0:
            reg_loss = reg_loss + lambda_normal * normal_consistency_loss(verts, faces)
        if lambda_vol > 0:
            vol_ratio = mesh_volume(verts, faces).abs() / vol_init
            reg_loss = reg_loss + lambda_vol * F.relu(vol_min_ratio - vol_ratio) ** 2

        total_loss = sil_loss + reg_loss
        total_loss.backward()
        optimizer.step()
        if scheduler:
            sched.step()

        # ── Logging ───────────────────────────────────────────────────
        if step % log_every == 0 or step == num_steps - 1:
            entry = {
                "step":     step,
                "loss":     total_loss.item(),
                "sil_loss": sil_loss.item(),
                "reg_loss": reg_loss.item(),
                "lr":       optimizer.param_groups[0]["lr"],
            }
            history.append(entry)
            print(
                f"  step {step:5d}/{num_steps} | "
                f"total={total_loss.item():.4f}  "
                f"sil={sil_loss.item():.4f}  "
                f"reg={reg_loss.item():.4f}  "
                f"lr={entry['lr']:.5f}"
            )

    return verts.detach(), history
