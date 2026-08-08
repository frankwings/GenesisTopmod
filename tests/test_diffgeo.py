"""
tests/test_diffgeo.py — differentiable geometry (topmod/diffgeo.py).

Groups
------
TestOracleParity     — torch forward == existing float implementation
                       (positions AND face rings) for every supported op
TestGradients        — gradcheck / gradient-flow for representative ops
TestCrust            — nonlinear crust path: parity + thickness gradient
TestDiffSequence     — composed sequences: parity, gradient flow, export
TestLinearityGuard   — nonlinear ops are rejected by the tracer

All CPU, float64.  The float implementations are the ground truth: the
torch path must reproduce them to tight tolerance.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from topmod import (
    make_cube, make_tetrahedron, make_icosahedron,
    create_crust,
)
from topmod.primitives import _build_mesh
from topmod.diffgeo import (
    LINEAR_OPS, NONLINEAR_OPS,
    mesh_to_arrays, trace_op, DiffSequence,
    _LINEAR_FLOAT_OPS, _NonLinearTrace,
)

TRIANGLE_ONLY = {"LOOP", "SQRT3"}

BASES = {
    "cube": make_cube,
    "tetrahedron": make_tetrahedron,
    "icosahedron": make_icosahedron,
}


def _float_reference(op_name, positions, faces, **params):
    """Run the existing float implementation and export the result."""
    mesh = _build_mesh([tuple(p) for p in positions],
                       [list(r) for r in faces])
    fn = _LINEAR_FLOAT_OPS[op_name]
    out = fn(mesh, **params)
    if out is None:
        out = mesh
    return mesh_to_arrays(out)


def _bases_for(op_name):
    if op_name in TRIANGLE_ONLY:
        return ["tetrahedron", "icosahedron"]
    return ["cube", "tetrahedron", "icosahedron"]


# ─────────────────────────────────────────────────────────────────────────────
# Oracle parity: torch path == float path
# ─────────────────────────────────────────────────────────────────────────────

class TestOracleParity:

    @pytest.mark.parametrize("op_name", LINEAR_OPS)
    def test_positions_and_faces_match_float_path(self, op_name):
        for base_name in _bases_for(op_name):
            positions, faces = mesh_to_arrays(BASES[base_name]())

            ref_pos, ref_faces = _float_reference(op_name, positions, faces)

            op = trace_op(op_name, len(positions), faces)
            verts = torch.tensor(positions, dtype=torch.float64)
            out = op.apply(verts)

            assert out.shape == (len(ref_pos), 3), (op_name, base_name)
            ref = torch.tensor(ref_pos, dtype=torch.float64)
            assert torch.allclose(out, ref, atol=1e-9), (op_name, base_name)
            assert op.faces == ref_faces, (op_name, base_name)

    @pytest.mark.parametrize("op_name,params", [
        ("VC",    {"offset": 0.3}),
        ("CCUT",  {"alpha": 0.7}),
        ("LSTYLE", {"length": 0.5}),
        ("PENT",  {"offset": 0.4}),
        ("PENT2", {"scale_factor": 0.6}),
        ("D1264", {"sf": 0.8}),
        ("ROOT4", {"a": 0.3, "twist": 0.2}),
        ("CHKB",  {"thickness": 0.3}),
        ("DSBC",  {"sf": 0.9, "length": 0.7}),
    ])
    def test_nondefault_params_match_float_path(self, op_name, params):
        positions, faces = mesh_to_arrays(make_cube())
        ref_pos, ref_faces = _float_reference(op_name, positions, faces,
                                              **params)
        op = trace_op(op_name, len(positions), faces, **params)
        out = op.apply(torch.tensor(positions, dtype=torch.float64))
        assert torch.allclose(out, torch.tensor(ref_pos, dtype=torch.float64),
                              atol=1e-9)
        assert op.faces == ref_faces

    def test_trace_is_topology_only(self):
        """The traced matrix depends on topology, not on the positions —
        applying it to *different* positions must equal the float op run
        on those positions."""
        positions, faces = mesh_to_arrays(make_cube())
        op = trace_op("CC", len(positions), faces)

        stretched = [(3.0 * x, 0.5 * y, z + 1.0) for x, y, z in positions]
        ref_pos, _ = _float_reference("CC", stretched, faces)
        out = op.apply(torch.tensor(stretched, dtype=torch.float64))
        assert torch.allclose(out, torch.tensor(ref_pos, dtype=torch.float64),
                              atol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# Gradients
# ─────────────────────────────────────────────────────────────────────────────

class TestGradients:

    @pytest.mark.parametrize("op_name", LINEAR_OPS)
    def test_gradcheck_on_tetrahedron(self, op_name):
        positions, faces = mesh_to_arrays(make_tetrahedron())
        op = trace_op(op_name, len(positions), faces)
        verts = torch.tensor(positions, dtype=torch.float64,
                             requires_grad=True)
        assert torch.autograd.gradcheck(op.apply, (verts,), atol=1e-6)

    def test_gradient_values_are_finite_and_nonzero(self):
        positions, faces = mesh_to_arrays(make_cube())
        op = trace_op("CC", len(positions), faces)
        verts = torch.tensor(positions, dtype=torch.float64,
                             requires_grad=True)
        loss = (op.apply(verts) ** 2).sum()
        loss.backward()
        assert verts.grad is not None
        assert torch.isfinite(verts.grad).all()
        assert verts.grad.abs().sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Crust (nonlinear)
# ─────────────────────────────────────────────────────────────────────────────

class TestCrust:

    @pytest.mark.parametrize("base_name", ["cube", "tetrahedron",
                                           "icosahedron"])
    @pytest.mark.parametrize("thickness", [0.1, 0.25, -0.2])
    def test_matches_float_path(self, base_name, thickness):
        mesh = BASES[base_name]()
        positions, faces = mesh_to_arrays(mesh)

        ref_mesh, _pairs = create_crust(BASES[base_name](),
                                        thickness=thickness)
        ref_pos, ref_faces = mesh_to_arrays(ref_mesh)

        op = trace_op("CRUST", len(positions), faces, thickness=thickness)
        out = op.apply(torch.tensor(positions, dtype=torch.float64))

        assert out.shape == (len(ref_pos), 3)
        assert torch.allclose(out, torch.tensor(ref_pos, dtype=torch.float64),
                              atol=1e-9)
        assert op.faces == ref_faces

    def test_thickness_gradient_flows(self):
        positions, faces = mesh_to_arrays(make_cube())
        t = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
        op = trace_op("CRUST", len(positions), faces, thickness=t)
        verts = torch.tensor(positions, dtype=torch.float64,
                             requires_grad=True)
        loss = (op.apply(verts) ** 2).sum()
        loss.backward()
        assert t.grad is not None and torch.isfinite(t.grad)
        assert abs(float(t.grad)) > 0
        assert verts.grad is not None and torch.isfinite(verts.grad).all()

    def test_gradcheck_wrt_positions(self):
        positions, faces = mesh_to_arrays(make_tetrahedron())
        op = trace_op("CRUST", len(positions), faces, thickness=0.15)
        verts = torch.tensor(positions, dtype=torch.float64,
                             requires_grad=True)
        assert torch.autograd.gradcheck(op.apply, (verts,), atol=1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Sequences
# ─────────────────────────────────────────────────────────────────────────────

class TestDiffSequence:

    def test_two_op_sequence_matches_float_path(self):
        seq = DiffSequence("cube").append("DS").append("CC")
        out = seq.forward()

        # float reference: DS then CC via the real implementations
        positions, faces = mesh_to_arrays(make_cube())
        mid_pos, mid_faces = _float_reference("DS", positions, faces)
        ref_pos, ref_faces = _float_reference("CC", mid_pos, mid_faces)

        assert torch.allclose(out.detach(),
                              torch.tensor(ref_pos, dtype=torch.float64),
                              atol=1e-9)
        assert seq.faces == ref_faces

    def test_gradient_reaches_base_vertices(self):
        seq = DiffSequence("cube").append("DS").append("CC")
        loss = (seq.forward() ** 2).sum()
        loss.backward()
        g = seq.verts0.grad
        assert g is not None
        assert torch.isfinite(g).all()
        assert g.abs().sum() > 0

    def test_sequence_with_crust_gradient(self):
        seq = DiffSequence("cube").append("CC").append("CRUST",
                                                       thickness=0.1)
        loss = (seq.forward() ** 2).sum()
        loss.backward()
        assert seq.verts0.grad is not None
        assert torch.isfinite(seq.verts0.grad).all()

    def test_triangles_export(self):
        seq = DiffSequence("cube").append("CC")
        tris = seq.triangles()
        assert tris.dtype == torch.long
        assert tris.shape[1] == 3
        # CC on cube: 24 quads -> 48 triangles
        assert tris.shape[0] == 48
        n_verts = seq.forward().shape[0]
        assert int(tris.max()) < n_verts
        assert int(tris.min()) >= 0

    def test_element_counts_against_closed_form(self):
        # CC on cube: V' = V+E+F = 8+12+6 = 26, F' = 2E = 24
        seq = DiffSequence("cube").append("CC")
        assert seq.forward().shape[0] == 26
        assert len(seq.faces) == 24

    def test_alternate_positions_forward(self):
        seq = DiffSequence("cube").append("SIMP")
        v2 = (seq.verts0.detach() * 2.0).requires_grad_(True)
        out = seq.forward(v2)
        loss = out.sum()
        loss.backward()
        assert v2.grad is not None


# ─────────────────────────────────────────────────────────────────────────────
# Tracer linearity guard
# ─────────────────────────────────────────────────────────────────────────────

class TestLinearityGuard:

    def test_unsupported_op_raises(self):
        positions, faces = mesh_to_arrays(make_cube())
        with pytest.raises(ValueError):
            trace_op("DOME", len(positions), faces)

    def test_star_not_silently_traced(self):
        """STAR (normal displacement) must not be in the linear registry."""
        assert "STAR" not in LINEAR_OPS
        assert "FRAC" not in LINEAR_OPS
        assert "DOME" not in LINEAR_OPS
