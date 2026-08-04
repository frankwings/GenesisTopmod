"""
manifold_loss.py — Differentiable manifold constraint losses for PyTorch.

Designed for integration with LATO.2 / DMesh-style mesh generation methods that
produce a (verts, faces) pair plus optional per-face existence probabilities.

Background
----------
A valid closed orientable 2-manifold satisfies three invariants:
  1. Edge manifold: every edge is adjacent to exactly 2 faces.
  2. Euler characteristic: V - E + F = 2C - 2g  (C components, g genus).
  3. Orientation consistency: every pair of adjacent faces share the edge in
     opposite traversal directions.

All three invariants are reformulated here as differentiable loss functions.

Differentiability
-----------------
Losses are differentiable w.r.t.:
  - ``face_probs`` [F]: per-face soft existence weights (primary gradient signal
    for topology optimization).
  - ``verts`` [V, 3]: vertex positions (gradient is zero for purely topological
    losses; non-zero only for ``edge_manifold_loss`` with normal weighting).

Typical loss values
-------------------
For a perfect manifold mesh with no face_probs:
  edge_manifold_loss → 0
  euler_loss         → 0
  orientation_loss   → 0

Usage
-----
>>> import torch
>>> from pipeline.manifold_loss import manifold_loss
>>> verts = torch.randn(8, 3, requires_grad=True)
>>> faces = torch.tensor([[0,1,2],[1,3,2],[...]], dtype=torch.int64)
>>> face_probs = torch.sigmoid(torch.randn(len(faces), requires_grad=True))
>>> loss = manifold_loss(verts, faces, face_probs=face_probs, target_genus=0)
>>> loss.backward()
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_edge_tables(
    faces: torch.Tensor,
    V:     int,
) -> Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Build undirected-edge lookup tables from a face index tensor.

    Parameters
    ----------
    faces : [F, 3]  int64  — face vertex indices.
    V     : int             — total number of vertices.

    Returns
    -------
    E          : int          — number of unique undirected edges.
    inv_idx    : [3F] int64  — maps each directed half-edge → unique edge index.
    half_sign  : [3F] float  — +1 if half-edge is in canonical direction, -1 otherwise.
    he_face_idx: [3F] int64  — which face index each half-edge belongs to.
    """
    F_count = faces.shape[0]
    device  = faces.device
    dtype   = torch.int64

    v0 = faces[:, 0].to(dtype)   # [F]
    v1 = faces[:, 1].to(dtype)   # [F]
    v2 = faces[:, 2].to(dtype)   # [F]

    # All directed half-edges: (v0→v1), (v1→v2), (v2→v0) for each face
    src = torch.cat([v0, v1, v2], dim=0)  # [3F]
    dst = torch.cat([v1, v2, v0], dim=0)  # [3F]

    # Canonical undirected edge: (min, max) pair
    emin = torch.minimum(src, dst)          # [3F]
    emax = torch.maximum(src, dst)          # [3F]

    # Hash to a single integer for torch.unique
    edge_hash = emin * V + emax            # [3F]  unique if V > max_vertex_id

    unique_hashes, inv_idx = torch.unique(edge_hash, return_inverse=True)
    E = unique_hashes.shape[0]

    # Sign: +1 if half-edge goes in canonical direction (src ≤ dst)
    half_sign = torch.where(
        src <= dst,
        torch.ones(3 * F_count, device=device, dtype=torch.float32),
        torch.full((3 * F_count,), -1.0, device=device, dtype=torch.float32),
    )

    # Which face does each half-edge belong to?
    # NOTE: torch.cat([v0, v1, v2]) produces COLUMN-CONSECUTIVE layout:
    #   positions 0..F-1 → face 0..F-1's (col0→col1) half-edges
    #   positions F..2F-1 → face 0..F-1's (col1→col2) half-edges
    #   positions 2F..3F-1 → face 0..F-1's (col2→col0) half-edges
    # repeat(3) = [0,1,..,F-1, 0,1,..,F-1, 0,1,..,F-1] matches this layout.
    # (repeat_interleave(3) would give [0,0,0, 1,1,1, ...] which is WRONG here.)
    he_face_idx = torch.arange(F_count, device=device, dtype=dtype).repeat(3)

    return E, inv_idx, half_sign, he_face_idx


def _get_face_probs(
    face_probs: Optional[torch.Tensor],
    F:          int,
    device:     torch.device,
    dtype:      torch.dtype,
) -> torch.Tensor:
    """Return face_probs or an all-ones tensor of shape [F] if None."""
    if face_probs is None:
        return torch.ones(F, device=device, dtype=dtype)
    return face_probs.to(dtype=dtype, device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Loss 1 — Edge Manifold Loss
# ─────────────────────────────────────────────────────────────────────────────

def edge_manifold_loss(
    verts:      torch.Tensor,
    faces:      torch.Tensor,
    face_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalise edges that are adjacent to ≠ 2 faces.

    For a valid closed 2-manifold every edge must be shared by exactly two
    faces.  When ``face_probs`` is provided (DMesh/LATO.2 style), each face
    contributes its probability to the adjacent-face count of every one of its
    edges.

    Parameters
    ----------
    verts      : [V, 3] float — vertex positions (grad flows; value not used).
    faces      : [F, 3] int  — face vertex indices.
    face_probs : [F] float, optional — per-face existence probabilities in
                  [0, 1].  If None, hard mesh (all faces exist with prob=1).

    Returns
    -------
    Scalar loss: mean of  (adjacent_face_count_e - 2)²  over all edges.
    For a perfect manifold mesh with no face_probs the loss is **exactly 0**.

    Gradients flow through ``face_probs``.
    """
    F_count = faces.shape[0]
    V       = verts.shape[0]
    device  = verts.device
    dtype   = verts.dtype

    fp = _get_face_probs(face_probs, F_count, device, dtype)

    E, inv_idx, _, he_face_idx = _build_edge_tables(faces, V)

    # Probability of each half-edge = probability of its parent face
    fp_rep = fp[he_face_idx]                                    # [3F]

    # Sum face probabilities per unique undirected edge
    edge_counts = torch.zeros(E, device=device, dtype=dtype)
    edge_counts = edge_counts.scatter_add(0, inv_idx, fp_rep)   # [E]

    # Penalty: (count - 2)^2 per edge
    # Anchor verts in the computation graph so backward() is safe even when
    # verts.requires_grad=True.  Topological losses don't depend on positions;
    # the added term contributes 0.0 numerically but gives verts a grad_fn.
    loss = ((edge_counts - 2.0) ** 2).mean() + 0.0 * verts.sum()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Loss 2 — Euler Characteristic Constraint Loss
# ─────────────────────────────────────────────────────────────────────────────

def euler_loss(
    verts:             torch.Tensor,
    faces:             torch.Tensor,
    target_genus:      int = 0,
    target_components: int = 1,
    face_probs:        Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalise deviation from the target Euler characteristic.

    For a closed orientable surface with *c* connected components and genus *g*:
        χ  =  V − E + F  =  2c − 2g

    When ``face_probs`` is provided the effective counts are:
        F_eff = Σ p_f
        E_eff = Σ_e clamp(Σ_{f∋e} p_f,  0, 1)
        V_eff = Σ_v clamp(Σ_{f∋v} p_f,  0, 1)

    The clamp ensures that an edge/vertex "exists" with weight 1 as soon as any
    adjacent face has non-zero probability.

    Parameters
    ----------
    verts             : [V, 3] float.
    faces             : [F, 3] int.
    target_genus      : int — desired topological genus (default 0 = sphere-like).
    target_components : int — desired number of connected components (default 1).
    face_probs        : [F] float, optional.

    Returns
    -------
    Scalar loss: (χ_eff - χ_target)².
    Gradients flow through ``face_probs``.
    """
    F_count = faces.shape[0]
    V       = verts.shape[0]
    device  = verts.device
    dtype   = verts.dtype

    target_chi = float(2 * target_components - 2 * target_genus)
    fp = _get_face_probs(face_probs, F_count, device, dtype)

    E, inv_idx, _, he_face_idx = _build_edge_tables(faces, V)

    fp_rep = fp[he_face_idx]        # [3F]

    # Effective face count
    F_eff = fp.sum()                # scalar

    # Effective edge count: clamp sum of adjacent face probs at 1
    edge_face_sum = torch.zeros(E, device=device, dtype=dtype).scatter_add(
        0, inv_idx, fp_rep
    )
    E_eff = edge_face_sum.clamp(max=1.0).sum()

    # Effective vertex count: clamp sum of adjacent face probs at 1
    all_verts_idx = faces.reshape(-1)    # [3F]
    vert_face_sum = torch.zeros(V, device=device, dtype=dtype).scatter_add(
        0, all_verts_idx.long(), fp_rep
    )
    V_eff = vert_face_sum.clamp(max=1.0).sum()

    chi_eff = V_eff - E_eff + F_eff
    return (chi_eff - target_chi) ** 2 + 0.0 * verts.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Loss 3 — Face Orientation Consistency Loss
# ─────────────────────────────────────────────────────────────────────────────

def orientation_consistency_loss(
    verts:      torch.Tensor,
    faces:      torch.Tensor,
    face_probs: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Penalise orientation-inconsistent adjacent face pairs.

    For a consistently oriented mesh, each undirected edge {u, v} should appear
    as (u→v) in exactly one adjacent face and as (v→u) in the other.  Formally,
    for each undirected edge e:

        signed_count(e)  =  Σ_{f∋e} p_f · sign(f, e)

    where sign(f, e) = +1 if face f traverses e in the canonical direction
    (lower index → higher index) and -1 otherwise.

    For a consistently oriented manifold:  signed_count(e) = 0  for every e.
    (The +1 and -1 contributions cancel.)

    Loss:  mean( signed_count(e)² )

    Parameters
    ----------
    verts      : [V, 3] float — not used in computation but enables a unified
                  function signature so loss.backward() is safe when verts has
                  requires_grad=True.
    faces      : [F, 3] int.
    face_probs : [F] float, optional.

    Returns
    -------
    Scalar loss.  Gradients flow through ``face_probs``.
    For a perfect consistently-oriented mesh the loss is **exactly 0**.
    """
    F_count = faces.shape[0]
    V       = verts.shape[0]
    device  = verts.device
    dtype   = verts.dtype

    fp = _get_face_probs(face_probs, F_count, device, dtype)

    E, inv_idx, half_sign, he_face_idx = _build_edge_tables(faces, V)

    fp_rep = fp[he_face_idx]    # [3F]

    # Signed count: sum of (sign × face_prob) per undirected edge
    signed_counts = torch.zeros(E, device=device, dtype=dtype).scatter_add(
        0, inv_idx, half_sign * fp_rep
    )

    return (signed_counts ** 2).mean() + 0.0 * verts.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Loss 4 — Combined Manifold Loss
# ─────────────────────────────────────────────────────────────────────────────

def manifold_loss(
    verts:        torch.Tensor,
    faces:        torch.Tensor,
    face_probs:   Optional[torch.Tensor] = None,
    target_genus: int   = 0,
    lambda_edge:  float = 1.0,
    lambda_euler: float = 0.5,
    lambda_orient: float = 0.3,
) -> torch.Tensor:
    """
    Weighted combination of the three manifold constraint losses.

    Parameters
    ----------
    verts        : [V, 3] float.
    faces        : [F, 3] int.
    face_probs   : [F] float, optional.
    target_genus : int (default 0).
    lambda_edge  : float — weight for edge-manifold loss (default 1.0).
    lambda_euler : float — weight for Euler characteristic loss (default 0.5).
    lambda_orient: float — weight for orientation consistency loss (default 0.3).

    Returns
    -------
    Scalar combined loss.
    """
    l_edge   = edge_manifold_loss(verts, faces, face_probs)
    l_euler  = euler_loss(verts, faces, target_genus=target_genus,
                          target_components=1, face_probs=face_probs)
    l_orient = orientation_consistency_loss(verts, faces, face_probs)

    return lambda_edge * l_edge + lambda_euler * l_euler + lambda_orient * l_orient


# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic utility
# ─────────────────────────────────────────────────────────────────────────────

def manifold_loss_breakdown(
    verts:        torch.Tensor,
    faces:        torch.Tensor,
    face_probs:   Optional[torch.Tensor] = None,
    target_genus: int   = 0,
    lambda_edge:  float = 1.0,
    lambda_euler: float = 0.5,
    lambda_orient: float = 0.3,
) -> dict:
    """
    Return a dict with individual and combined loss values (no gradients).

    Useful for monitoring training progress without modifying the compute graph.
    """
    with torch.no_grad():
        l_edge   = edge_manifold_loss(verts, faces, face_probs).item()
        l_euler  = euler_loss(verts, faces, target_genus=target_genus,
                              target_components=1, face_probs=face_probs).item()
        l_orient = orientation_consistency_loss(verts, faces, face_probs).item()
        total    = (lambda_edge * l_edge +
                    lambda_euler * l_euler +
                    lambda_orient * l_orient)
    return {
        "edge_manifold":  l_edge,
        "euler":          l_euler,
        "orientation":    l_orient,
        "total":          total,
    }
