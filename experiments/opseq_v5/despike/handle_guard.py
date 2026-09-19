#!/usr/bin/env python3
"""handle_guard.py — persistent-homology-inspired anti-collapse guard for a DLFL-added handle (borrow #1 from
Gu et al. ICASSP 2026). A handle dies when its tunnel THROAT (the narrowest cross-section of the hole) closes; the
throat width is a cheap differentiable proxy for the H1 persistence of that loop. Keep it open -> the handle survives
the post-add DR loop (Laplacian + collapse remesh) that would otherwise pinch a thin handle shut (mug-handle failure).

throat_openness_loss(V, a0, u, tube_verts, target, soft): for the tunnel with centreline point a0 and axis u, the hole
radius at a vertex = its distance to the axis line. The throat = the soft-min of these radii over the tube ring vertices.
Penalise (target - throat)_+ so gradient descent pushes the narrowest ring outward, keeping the hole open. Differentiable,
O(#tube_verts), no PH library."""
import torch

def axis_radius(V, a0, u):
    """distance of each vertex to the infinite line (a0, unit u)."""
    d = V - a0                                  # [N,3]
    proj = (d @ u).unsqueeze(-1) * u            # component along axis
    return (d - proj).norm(dim=-1)              # perpendicular distance = hole radius at that vertex

def throat_openness_loss(V, a0, u, tube_idx, target, soft=40.0):
    """soft-min radius over the tube vertices, penalised below `target`. soft = sharpness of the min."""
    r = axis_radius(V[tube_idx], a0, u)         # radii of the tube ring
    throat = -torch.logsumexp(-soft * r, 0) / soft   # smooth min
    return torch.relu(target - throat), throat
