"""
tests/test_manifold_loss.py — Comprehensive tests for pipeline/manifold_loss.py.

Tests are grouped into:
    TestEdgeManifoldLoss       — Loss 1 (edge adjacency count)
    TestEulerLoss              — Loss 2 (Euler characteristic)
    TestOrientationLoss        — Loss 3 (orientation consistency)
    TestManifoldLoss           — Combined loss
    TestGradientFlow           — .backward() for all losses
    TestFaceProbsSoft          — Soft (probabilistic) face_probs variants
    TestBreakdownUtility       — manifold_loss_breakdown() helper
    TestEdgeCases              — Single-triangle, degenerate, large meshes

Canonical meshes used
---------------------
    tetrahedron : 4 V, 4 F, 6 E  — valid genus-0 closed manifold
    octahedron  : 6 V, 8 F, 12 E — valid genus-0 closed manifold
    two-triangles (non-manifold) : share an edge, no closing faces — open boundary
    extra-face mesh              : tetrahedron + one extra face on a shared edge
                                   → that edge now has 3 adjacent faces
    flipped-face mesh            : tetrahedron with one face winding reversed
                                   → orientation inconsistency
"""

from __future__ import annotations

import math
import sys
import os

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.manifold_loss import (
    edge_manifold_loss,
    euler_loss,
    orientation_consistency_loss,
    manifold_loss,
    manifold_loss_breakdown,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared mesh fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _tet():
    """Tetrahedron: valid genus-0 manifold.  4V, 4F, 6E.  χ = V−E+F = 2."""
    verts = torch.tensor([
        [ 1.0,  1.0,  1.0],
        [ 1.0, -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0, 1, 2],
        [0, 2, 3],
        [0, 3, 1],
        [1, 3, 2],
    ], dtype=torch.int64)
    return verts, faces


def _oct():
    """Octahedron: valid genus-0 manifold.  6V, 8F, 12E.  χ = 2."""
    verts = torch.tensor([
        [ 1.0,  0.0,  0.0],
        [-1.0,  0.0,  0.0],
        [ 0.0,  1.0,  0.0],
        [ 0.0, -1.0,  0.0],
        [ 0.0,  0.0,  1.0],
        [ 0.0,  0.0, -1.0],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0, 2, 4], [0, 4, 3], [0, 3, 5], [0, 5, 2],
        [1, 4, 2], [1, 3, 4], [1, 5, 3], [1, 2, 5],
    ], dtype=torch.int64)
    return verts, faces


def _tet_with_extra_face():
    """
    Tetrahedron + one extra face [0, 1, 3] using ONLY existing vertices.

    The extra face reuses edges (0,1), (1,3) and (0,3) that already exist in
    the base tetrahedron.  Each of those three edges now has 3 adjacent faces
    instead of 2 → non-manifold violation.

    Key property: when face_probs[-1] is set to 0, ALL edges revert to count=2
    (no orphan edges), so edge_manifold_loss returns exactly 0.
    """
    verts, faces = _tet()
    # All three vertices already exist; no new vertex needed.
    extra_f = torch.tensor([[0, 1, 3]], dtype=torch.int64)
    faces2  = torch.cat([faces, extra_f], dim=0)
    return verts, faces2


def _tet_flipped():
    """Tetrahedron with face[0] winding reversed → orientation violation."""
    verts, faces = _tet()
    faces_f = faces.clone()
    # Reverse winding of first face: [0,1,2] → [0,2,1]
    faces_f[0, 1], faces_f[0, 2] = faces[0, 2].item(), faces[0, 1].item()
    return verts, faces_f


def _two_open_triangles():
    """
    Two triangles sharing edge (1,2) but no closing faces.
    Boundary mesh: some edges have only 1 adjacent face.
    """
    verts = torch.tensor([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.5, 1.0, 0.0],
        [0.5, -1.0, 0.0],
    ], dtype=torch.float32)
    faces = torch.tensor([
        [0, 1, 2],
        [0, 3, 1],
    ], dtype=torch.int64)
    return verts, faces


# ─────────────────────────────────────────────────────────────────────────────
# Loss 1 — Edge Manifold Loss
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeManifoldLoss:

    def test_tetrahedron_is_zero(self):
        """Perfect manifold → loss = 0."""
        verts, faces = _tet()
        loss = edge_manifold_loss(verts, faces)
        assert loss.item() == pytest.approx(0.0, abs=1e-6), \
            f"Tetrahedron edge manifold loss should be 0, got {loss.item():.6f}"

    def test_octahedron_is_zero(self):
        """Octahedron is valid manifold → loss = 0."""
        verts, faces = _oct()
        loss = edge_manifold_loss(verts, faces)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_extra_face_is_positive(self):
        """Adding a face that shares an existing edge → one edge has 3 faces → loss > 0."""
        verts, faces = _tet_with_extra_face()
        loss = edge_manifold_loss(verts, faces)
        assert loss.item() > 0.0, \
            f"Non-manifold mesh should have positive loss, got {loss.item()}"

    def test_extra_face_penalty_magnitude(self):
        """
        The extra face creates one edge with count=3 and possibly others with
        count=1 (boundary). (count-2)^2 = 1 for each.  The mean should be > 0.
        """
        verts, faces = _tet_with_extra_face()
        loss = edge_manifold_loss(verts, faces)
        # There must be at least one edge violating the count-2 rule
        assert loss.item() >= 1.0 / (faces.shape[0] * 3), \
            "Loss should reflect at least one violated edge"

    def test_open_boundary_mesh_positive(self):
        """Two open triangles have boundary edges (count=1) → loss > 0."""
        verts, faces = _two_open_triangles()
        loss = edge_manifold_loss(verts, faces)
        assert loss.item() > 0.0

    def test_returns_scalar_tensor(self):
        verts, faces = _tet()
        loss = edge_manifold_loss(verts, faces)
        assert loss.shape == torch.Size([])
        assert loss.dtype == torch.float32

    def test_with_all_ones_face_probs(self):
        """face_probs = all-1.0 should give same result as no face_probs."""
        verts, faces = _tet()
        fp   = torch.ones(faces.shape[0])
        l1   = edge_manifold_loss(verts, faces)
        l2   = edge_manifold_loss(verts, faces, face_probs=fp)
        assert l1.item() == pytest.approx(l2.item(), abs=1e-6)

    def test_face_probs_zero_removes_face(self):
        """Setting the extra face's prob to 0 should recover manifold (loss ≈ 0)."""
        verts, faces = _tet_with_extra_face()
        fp = torch.ones(faces.shape[0])
        fp[-1] = 0.0   # zero out the extra face
        loss = edge_manifold_loss(verts, faces, face_probs=fp)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_soft_face_probs_interpolation(self):
        """Intermediate face_probs give intermediate loss between 0 and hard-1 loss."""
        verts, faces = _tet_with_extra_face()
        fp_half = torch.ones(faces.shape[0])
        fp_half[-1] = 0.5
        l_half = edge_manifold_loss(verts, faces, face_probs=fp_half).item()
        l_full = edge_manifold_loss(verts, faces).item()
        assert 0.0 < l_half < l_full, \
            f"Soft prob=0.5 loss {l_half:.4f} should be between 0 and {l_full:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Loss 2 — Euler Loss
# ─────────────────────────────────────────────────────────────────────────────

class TestEulerLoss:

    def test_tetrahedron_genus0_is_zero(self):
        """Tetrahedron has χ=2 (genus=0, 1 component); euler_loss should be 0."""
        verts, faces = _tet()
        loss = euler_loss(verts, faces, target_genus=0, target_components=1)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_octahedron_genus0_is_zero(self):
        verts, faces = _oct()
        loss = euler_loss(verts, faces, target_genus=0, target_components=1)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_wrong_genus_target_positive(self):
        """Genus-0 mesh with target_genus=1 → χ_target=0 ≠ 2 → loss > 0."""
        verts, faces = _tet()
        loss = euler_loss(verts, faces, target_genus=1, target_components=1)
        # chi_eff = 2, target = 0, loss = (2-0)^2 = 4
        assert loss.item() == pytest.approx(4.0, abs=1e-4)

    def test_wrong_component_count_positive(self):
        """target_components=2 → χ_target=4; loss = (2-4)^2 = 4."""
        verts, faces = _tet()
        loss = euler_loss(verts, faces, target_genus=0, target_components=2)
        assert loss.item() == pytest.approx(4.0, abs=1e-4)

    def test_returns_scalar_tensor(self):
        verts, faces = _tet()
        loss = euler_loss(verts, faces)
        assert loss.shape == torch.Size([])

    def test_with_all_ones_face_probs_equals_hard(self):
        verts, faces = _tet()
        fp   = torch.ones(faces.shape[0])
        l1   = euler_loss(verts, faces)
        l2   = euler_loss(verts, faces, face_probs=fp)
        assert l1.item() == pytest.approx(l2.item(), abs=1e-4)

    def test_face_probs_zero_all_gives_zero_eff(self):
        """If all face probs → 0, V_eff=E_eff=F_eff=0 → chi_eff=0."""
        verts, faces = _tet()
        fp = torch.zeros(faces.shape[0])
        # chi_eff = 0, target_chi = 2 → loss = 4
        loss = euler_loss(verts, faces, target_genus=0, face_probs=fp)
        assert loss.item() == pytest.approx(4.0, abs=1e-4)

    def test_soft_euler_interpolates(self):
        """Partial face removal shifts chi_eff away from 2."""
        verts, faces = _tet()
        fp = torch.ones(faces.shape[0])
        fp[0] = 0.0    # remove one face
        loss = euler_loss(verts, faces, target_genus=0, face_probs=fp)
        # With one face removed, F_eff drops; chi_eff ≠ 2
        # The loss should be positive
        assert loss.item() > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Loss 3 — Orientation Consistency Loss
# ─────────────────────────────────────────────────────────────────────────────

class TestOrientationLoss:

    def test_tetrahedron_is_zero(self):
        """Consistently oriented tetrahedron → orientation loss = 0."""
        verts, faces = _tet()
        loss = orientation_consistency_loss(verts, faces)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_octahedron_is_zero(self):
        verts, faces = _oct()
        loss = orientation_consistency_loss(verts, faces)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_flipped_face_positive(self):
        """One reversed face → signed_count ≠ 0 for its edges → loss > 0."""
        verts, faces = _tet_flipped()
        loss = orientation_consistency_loss(verts, faces)
        assert loss.item() > 0.0, \
            f"Flipped-face mesh should have positive orientation loss, got {loss.item()}"

    def test_consistent_orientation_after_fix(self):
        """Fixing the flip should restore loss to 0."""
        verts, faces = _tet_flipped()
        # Un-flip: restore original winding for face 0
        faces_fixed = faces.clone()
        faces_fixed[0, 1], faces_fixed[0, 2] = faces[0, 2].item(), faces[0, 1].item()
        loss = orientation_consistency_loss(verts, faces_fixed)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_returns_scalar_tensor(self):
        verts, faces = _tet()
        loss = orientation_consistency_loss(verts, faces)
        assert loss.shape == torch.Size([])

    def test_with_all_ones_face_probs_equals_hard(self):
        verts, faces = _tet()
        fp = torch.ones(faces.shape[0])
        l1 = orientation_consistency_loss(verts, faces)
        l2 = orientation_consistency_loss(verts, faces, face_probs=fp)
        assert l1.item() == pytest.approx(l2.item(), abs=1e-6)

    def test_zero_prob_face_zeroes_contribution(self):
        """Setting flipped face prob to 0 removes its contribution."""
        verts, faces = _tet_flipped()
        fp = torch.ones(faces.shape[0])
        fp[0] = 0.0    # zero out the flipped face
        loss = orientation_consistency_loss(verts, faces, face_probs=fp)
        # With the flipped face removed, loss should drop but not necessarily 0
        # (remaining faces are consistently oriented among themselves)
        l_full = orientation_consistency_loss(verts, faces).item()
        assert loss.item() < l_full, \
            "Zeroing flipped face should reduce orientation loss"


# ─────────────────────────────────────────────────────────────────────────────
# Combined Loss
# ─────────────────────────────────────────────────────────────────────────────

class TestManifoldLoss:

    def test_good_mesh_near_zero(self):
        """Combined loss on tetrahedron should be very close to 0."""
        verts, faces = _tet()
        loss = manifold_loss(verts, faces, target_genus=0)
        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_bad_mesh_positive(self):
        """Mesh with extra face → combined loss > 0."""
        verts, faces = _tet_with_extra_face()
        loss = manifold_loss(verts, faces, target_genus=0)
        assert loss.item() > 0.0

    def test_weight_scaling(self):
        """Doubling lambda_edge should increase loss for a non-manifold mesh."""
        verts, faces = _tet_with_extra_face()
        l1 = manifold_loss(verts, faces, lambda_edge=1.0).item()
        l2 = manifold_loss(verts, faces, lambda_edge=2.0).item()
        assert l2 > l1, "Larger lambda_edge should give larger combined loss"

    def test_zero_weights_gives_zero(self):
        """All lambdas = 0 → loss = 0 regardless of mesh quality."""
        verts, faces = _tet_with_extra_face()
        loss = manifold_loss(verts, faces,
                             lambda_edge=0.0,
                             lambda_euler=0.0,
                             lambda_orient=0.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_individual_weights(self):
        """Manually weighted sum should equal the combined loss."""
        verts, faces = _tet_with_extra_face()
        le = 1.5; leu = 0.7; lo = 0.4
        l_combo = manifold_loss(verts, faces, lambda_edge=le,
                                lambda_euler=leu, lambda_orient=lo).item()
        l_manual = (
            le  * edge_manifold_loss(verts, faces).item() +
            leu * euler_loss(verts, faces).item() +
            lo  * orientation_consistency_loss(verts, faces).item()
        )
        assert l_combo == pytest.approx(l_manual, rel=1e-5)

    def test_with_face_probs(self):
        """Combined loss should work with face_probs."""
        verts, faces = _tet_with_extra_face()
        fp = torch.full((faces.shape[0],), 0.8)
        loss = manifold_loss(verts, faces, face_probs=fp)
        assert loss.item() >= 0.0

    def test_returns_scalar_tensor(self):
        verts, faces = _tet()
        loss = manifold_loss(verts, faces)
        assert loss.shape == torch.Size([])


# ─────────────────────────────────────────────────────────────────────────────
# Gradient flow tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGradientFlow:
    """Ensure .backward() works and produces finite, non-trivial gradients."""

    def _check_grad(self, loss: torch.Tensor, param: torch.Tensor, name: str):
        """Run backward and assert gradients are finite and non-zero."""
        if param.grad is not None:
            param.grad.zero_()
        loss.backward()
        grad = param.grad
        assert grad is not None,       f"No gradient for {name}"
        assert torch.isfinite(grad).all(), f"Non-finite gradient for {name}"
        assert grad.abs().sum().item() > 0.0, f"Zero gradient for {name}"

    # ── edge manifold loss ──────────────────────────────────────────────────

    def test_edge_loss_grad_face_probs(self):
        verts, faces = _tet_with_extra_face()
        fp = torch.full((faces.shape[0],), 0.9, requires_grad=True)
        loss = edge_manifold_loss(verts, faces, face_probs=fp)
        self._check_grad(loss, fp, "face_probs (edge_manifold_loss)")

    def test_edge_loss_grad_verts_no_error(self):
        """
        Gradient w.r.t. verts is zero (topological loss) but backward() must
        not raise, and verts.grad must exist (all zeros).
        """
        verts, faces = _tet_with_extra_face()
        verts_rg = verts.clone().requires_grad_(True)
        loss = edge_manifold_loss(verts_rg, faces)
        loss.backward()
        assert verts_rg.grad is not None, "verts.grad should exist (even if all zeros)"
        assert torch.allclose(verts_rg.grad, torch.zeros_like(verts_rg)), \
            "Gradient w.r.t. verts should be zero for a topological loss"

    # ── euler loss ─────────────────────────────────────────────────────────

    def test_euler_loss_grad_face_probs(self):
        verts, faces = _tet()
        # Use wrong genus to get non-zero loss (and non-zero gradients)
        fp = torch.full((faces.shape[0],), 0.9, requires_grad=True)
        loss = euler_loss(verts, faces, target_genus=1, face_probs=fp)
        self._check_grad(loss, fp, "face_probs (euler_loss)")

    def test_euler_loss_backward_no_error(self):
        verts, faces = _tet()
        verts_rg = verts.clone().requires_grad_(True)
        fp = torch.full((faces.shape[0],), 0.8, requires_grad=True)
        loss = euler_loss(verts_rg, faces, target_genus=0, face_probs=fp)
        loss.backward()   # must not raise

    # ── orientation loss ────────────────────────────────────────────────────

    def test_orient_loss_grad_face_probs(self):
        verts, faces = _tet_flipped()
        fp = torch.full((faces.shape[0],), 0.9, requires_grad=True)
        loss = orientation_consistency_loss(verts, faces, face_probs=fp)
        self._check_grad(loss, fp, "face_probs (orientation_loss)")

    def test_orient_loss_backward_no_error(self):
        """backward() must not raise even when only verts has requires_grad."""
        verts, faces = _tet_flipped()
        verts_rg = verts.clone().requires_grad_(True)
        loss = orientation_consistency_loss(verts_rg, faces)
        loss.backward()   # must not raise
        assert verts_rg.grad is not None
        assert torch.allclose(verts_rg.grad, torch.zeros_like(verts_rg))

    # ── combined loss ───────────────────────────────────────────────────────

    def test_combined_loss_grad_face_probs(self):
        verts, faces = _tet_with_extra_face()
        fp = torch.full((faces.shape[0],), 0.9, requires_grad=True)
        loss = manifold_loss(verts, faces, face_probs=fp, target_genus=0)
        self._check_grad(loss, fp, "face_probs (manifold_loss)")

    def test_combined_loss_grad_verts_no_error(self):
        """backward() must not raise; verts.grad exists (all zeros)."""
        verts, faces = _tet_with_extra_face()
        verts_rg = verts.clone().requires_grad_(True)
        fp = torch.full((faces.shape[0],), 0.9, requires_grad=True)
        loss = manifold_loss(verts_rg, faces, face_probs=fp)
        loss.backward()   # must not raise
        assert verts_rg.grad is not None
        assert torch.allclose(verts_rg.grad, torch.zeros_like(verts_rg)), \
            "Manifold loss has zero gradient w.r.t. vertex positions"

    def test_gradient_optimization_reduces_loss(self):
        """A few Adam steps on face_probs should reduce the manifold loss."""
        verts, faces = _tet_with_extra_face()
        logits = torch.zeros(faces.shape[0], requires_grad=True)
        optim  = torch.optim.Adam([logits], lr=0.1)

        losses = []
        for _ in range(20):
            optim.zero_grad()
            fp   = torch.sigmoid(logits)
            loss = manifold_loss(verts, faces, face_probs=fp)
            loss.backward()
            optim.step()
            losses.append(loss.item())

        assert losses[-1] < losses[0], \
            f"Optimization did not reduce loss: {losses[0]:.4f} → {losses[-1]:.4f}"


# ─────────────────────────────────────────────────────────────────────────────
# Soft face_probs detailed tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFaceProbsSoft:

    def test_monotone_in_rogue_prob(self):
        """
        As the rogue face prob increases from 0 to 1, edge manifold loss should
        increase monotonically (more of the non-manifold face 'exists').
        """
        verts, faces = _tet_with_extra_face()
        probs = [0.0, 0.25, 0.5, 0.75, 1.0]
        losses = []
        for p in probs:
            fp = torch.ones(faces.shape[0])
            fp[-1] = p
            losses.append(edge_manifold_loss(verts, faces, face_probs=fp).item())

        for i in range(len(losses) - 1):
            assert losses[i] <= losses[i + 1] + 1e-6, \
                f"Loss not monotone at p={probs[i]}: {losses[i]:.4f} > {losses[i+1]:.4f}"

    def test_all_zeros_edge_loss(self):
        """All face probs = 0 → every edge has count=0 → (0-2)^2 = 4 for all edges."""
        verts, faces = _tet()
        fp   = torch.zeros(faces.shape[0])
        loss = edge_manifold_loss(verts, faces, face_probs=fp).item()
        # Each of 6 edges: (0-2)^2 = 4; mean = 4
        assert loss == pytest.approx(4.0, abs=1e-5)

    def test_euler_face_probs_sum(self):
        """
        For tetrahedron with one face at prob 0.5:
        F_eff = 3.5; chi_eff ≠ 2 → positive loss.
        """
        verts, faces = _tet()
        fp = torch.ones(faces.shape[0])
        fp[0] = 0.5
        loss = euler_loss(verts, faces, target_genus=0, face_probs=fp)
        assert loss.item() > 0.0

    def test_orientation_soft_interpolation(self):
        """
        Flipped face at prob p: signed_count magnitude scales with p.
        """
        verts, faces = _tet_flipped()
        # Full prob for flipped face → positive loss
        fp_full = torch.ones(faces.shape[0])
        l_full  = orientation_consistency_loss(verts, faces, face_probs=fp_full).item()
        # Zero prob for flipped face → reduced loss
        fp_zero = torch.ones(faces.shape[0])
        fp_zero[0] = 0.0
        l_zero = orientation_consistency_loss(verts, faces, face_probs=fp_zero).item()
        assert l_full > l_zero

    def test_gradient_nonzero_wrt_nonmanifold_face(self):
        """
        The gradient of edge_manifold_loss w.r.t. face_probs[extra_face] should
        be non-zero, since that face participates in the violated edge.
        """
        verts, faces = _tet_with_extra_face()
        fp = torch.full((faces.shape[0],), 0.9, requires_grad=True)
        loss = edge_manifold_loss(verts, faces, face_probs=fp)
        loss.backward()
        # The gradient of the last (rogue) face should be largest
        grad = fp.grad.abs()
        assert grad[-1].item() > 0.0, \
            "Rogue face should have non-zero gradient in edge manifold loss"


# ─────────────────────────────────────────────────────────────────────────────
# manifold_loss_breakdown utility
# ─────────────────────────────────────────────────────────────────────────────

class TestBreakdownUtility:

    def test_returns_dict_with_expected_keys(self):
        verts, faces = _tet()
        result = manifold_loss_breakdown(verts, faces)
        for key in ("edge_manifold", "euler", "orientation", "total"):
            assert key in result, f"Key {key!r} missing from breakdown"

    def test_total_matches_weighted_sum(self):
        verts, faces = _tet_with_extra_face()
        le, leu, lo = 1.0, 0.5, 0.3
        bd = manifold_loss_breakdown(verts, faces,
                                     lambda_edge=le, lambda_euler=leu,
                                     lambda_orient=lo)
        expected = (le * bd["edge_manifold"] +
                    leu * bd["euler"]        +
                    lo  * bd["orientation"])
        assert bd["total"] == pytest.approx(expected, rel=1e-5)

    def test_no_grad_in_breakdown(self):
        """manifold_loss_breakdown runs under no_grad → no side effects."""
        verts, faces = _tet_with_extra_face()
        fp = torch.full((faces.shape[0],), 0.9, requires_grad=True)
        manifold_loss_breakdown(verts, faces, face_probs=fp)
        # After breakdown, fp.grad should still be None (no backward called)
        assert fp.grad is None

    def test_breakdown_values_are_floats(self):
        verts, faces = _tet()
        bd = manifold_loss_breakdown(verts, faces)
        for k, v in bd.items():
            assert isinstance(v, float), f"Value for {k} should be float, got {type(v)}"

    def test_good_mesh_all_zeros(self):
        verts, faces = _tet()
        bd = manifold_loss_breakdown(verts, faces, target_genus=0)
        assert bd["edge_manifold"] == pytest.approx(0.0, abs=1e-6)
        assert bd["euler"]        == pytest.approx(0.0, abs=1e-6)
        assert bd["orientation"]  == pytest.approx(0.0, abs=1e-6)
        assert bd["total"]        == pytest.approx(0.0, abs=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:

    def test_single_triangle(self):
        """Single triangle: every edge has count=1, not 2 → edge loss > 0."""
        verts = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces = torch.tensor([[0,1,2]])
        loss = edge_manifold_loss(verts, faces)
        # All 3 edges have count=1; (1-2)^2 = 1 each; mean = 1
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_single_triangle_euler(self):
        """Single triangle: V=3, E=3, F=1, chi=1; target=2 → euler loss = 1."""
        verts = torch.tensor([[0.,0.,0.],[1.,0.,0.],[0.,1.,0.]])
        faces = torch.tensor([[0,1,2]])
        loss = euler_loss(verts, faces, target_genus=0, target_components=1)
        # chi_eff = 3 - 3 + 1 = 1; target = 2; loss = (1-2)^2 = 1
        assert loss.item() == pytest.approx(1.0, abs=1e-5)

    def test_larger_mesh_octahedron_stays_zero(self):
        """Octahedron: all three losses should be exactly 0."""
        verts, faces = _oct()
        bd = manifold_loss_breakdown(verts, faces, target_genus=0)
        assert bd["edge_manifold"] == pytest.approx(0.0, abs=1e-6)
        assert bd["euler"]        == pytest.approx(0.0, abs=1e-6)
        assert bd["orientation"]  == pytest.approx(0.0, abs=1e-6)

    def test_face_probs_in_range_zero_one(self):
        """face_probs outside [0,1] still runs without error (unclamped)."""
        verts, faces = _tet()
        fp = torch.full((faces.shape[0],), 1.5)   # probabilities > 1
        loss = edge_manifold_loss(verts, faces, face_probs=fp)
        assert torch.isfinite(loss)

    def test_cuda_if_available(self):
        """All losses should work on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        verts, faces = _tet()
        verts_c = verts.cuda()
        faces_c = faces.cuda()
        for fn in [edge_manifold_loss, orientation_consistency_loss]:
            loss = fn(verts_c, faces_c)
            assert loss.device.type == "cuda"
            assert torch.isfinite(loss)
        loss_e = euler_loss(verts_c, faces_c)
        assert loss_e.device.type == "cuda"

    def test_determinism(self):
        """Calling twice gives identical results."""
        verts, faces = _tet_with_extra_face()
        l1 = manifold_loss(verts, faces).item()
        l2 = manifold_loss(verts, faces).item()
        assert l1 == l2

    def test_dtype_float64(self):
        """Losses work with float64 inputs."""
        verts, faces = _tet()
        verts64 = verts.double()
        loss = edge_manifold_loss(verts64, faces)
        assert loss.dtype == torch.float64
        assert loss.item() == pytest.approx(0.0, abs=1e-12)

    def test_all_losses_finite_on_degenerate_probs(self):
        """face_probs = 0 or 1 should give finite losses (no NaN/Inf)."""
        verts, faces = _tet_with_extra_face()
        for val in [0.0, 1.0]:
            fp = torch.full((faces.shape[0],), val)
            for fn in [edge_manifold_loss, euler_loss, orientation_consistency_loss]:
                loss = fn(verts, faces, face_probs=fp)
                assert torch.isfinite(loss), \
                    f"{fn.__name__} returned non-finite value for face_probs={val}"
