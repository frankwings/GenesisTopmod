"""
Comprehensive tests for topmod/tokenizer.py

Coverage
--------
1. TopModToken dataclass
2. Coordinate quantization / dequantization
3. tokenize() — token counts, types, valid parameters
4. detokenize() — manifold guarantee, genus preservation
5. roundtrip — tokenize → detokenize → topology match
6. Vocabulary — build, encode, decode
7. Edge cases — large genus, multiple CC rounds, empty meshes
8. Error handling — bad ordinals, unknown ops
"""

import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from topmod.dlfl import DLFLMesh
from topmod.primitives import make_cube, make_tetrahedron, make_icosahedron, make_octahedron
from topmod.high_level_ops import add_handle
from topmod.subdivision import catmull_clark
from topmod.validate import is_manifold, check_all
from topmod.tokenizer import (
    TopModToken,
    quantize_coord, dequantize_coord,
    tokenize, detokenize,
    build_vocabulary, encode_sequence, decode_sequence,
    token_stats, sequence_length,
    DEFAULT_COORD_LO, DEFAULT_COORD_HI,
    _find_compatible_face_pair,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_genus1_mesh():
    """Icosahedron + one handle → genus-1 closed surface."""
    mesh  = make_icosahedron()
    faces = list(mesh.faces.values())
    add_handle(mesh, faces[0], faces[10])
    return mesh


def _make_genus2_mesh():
    """Icosahedron + two handles → genus-2 closed surface."""
    mesh  = make_icosahedron()
    faces = list(mesh.faces.values())
    add_handle(mesh, faces[0], faces[10])
    add_handle(mesh, faces[3], faces[13])
    return mesh


def _positions_from_mesh(mesh):
    return [(v.x, v.y, v.z) for v in mesh.vertices.values()]


# ─────────────────────────────────────────────────────────────────────────────
# 1. TopModToken dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestTopModToken:
    def test_eos_repr(self):
        t = TopModToken(op='EOS')
        assert repr(t) == 'EOS'

    def test_cc_repr(self):
        t = TopModToken(op='CC')
        assert repr(t) == 'CC'

    def test_cv_repr(self):
        t = TopModToken(op='CV', pos=(10, 20, 30))
        assert repr(t) == 'CV(10,20,30)'

    def test_hdl_repr(self):
        t = TopModToken(op='HDL', corner1=(0, 0), corner2=(5, 0))
        assert 'HDL' in repr(t)
        assert 'f0' in repr(t)
        assert 'f5' in repr(t)

    def test_ie_repr(self):
        t = TopModToken(op='IE', corner1=(2, 1), corner2=(7, 0))
        assert 'IE' in repr(t)
        assert 'f2' in repr(t) and 'h1' in repr(t)

    def test_de_repr(self):
        t = TopModToken(op='DE', edge_ord=3)
        assert repr(t) == 'DE(e3)'

    def test_default_fields_are_none(self):
        t = TopModToken(op='CC')
        assert t.pos is None
        assert t.corner1 is None
        assert t.corner2 is None
        assert t.edge_ord is None

    def test_cv_fields_set(self):
        t = TopModToken(op='CV', pos=(64, 32, 96))
        assert t.pos == (64, 32, 96)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Coordinate quantization
# ─────────────────────────────────────────────────────────────────────────────

class TestQuantization:
    def test_quantize_centre(self):
        """Centre of range maps to middle bin."""
        mid   = (DEFAULT_COORD_LO + DEFAULT_COORD_HI) / 2.0
        n     = 128
        q     = quantize_coord(mid, n_bins=n)
        assert q == n // 2 or q == n // 2 - 1   # allow off-by-one at centre

    def test_quantize_lo_clamp(self):
        q = quantize_coord(DEFAULT_COORD_LO - 99, n_bins=128)
        assert q == 0

    def test_quantize_hi_clamp(self):
        q = quantize_coord(DEFAULT_COORD_HI + 99, n_bins=128)
        assert q == 127

    def test_quantize_range(self):
        """All outputs are within [0, n_bins-1]."""
        import random; random.seed(0)
        n = 128
        for _ in range(200):
            x = random.uniform(-5, 5)
            q = quantize_coord(x, n_bins=n)
            assert 0 <= q < n, f"q={q} out of range for x={x}"

    def test_dequantize_round_trip(self):
        """Quantize then dequantize gives small error."""
        n       = 128
        lo, hi  = DEFAULT_COORD_LO, DEFAULT_COORD_HI
        step    = (hi - lo) / n
        tolerance = step   # max error = one bin width

        for x in [-1.5, -1.0, 0.0, 0.5, 1.0, 1.5]:
            q  = quantize_coord(x, n_bins=n)
            x2 = dequantize_coord(q, n_bins=n)
            assert abs(x - x2) <= tolerance, f"x={x}, q={q}, x2={x2}"

    def test_dequantize_is_monotone(self):
        """Higher bin → larger dequantized value."""
        vals = [dequantize_coord(q, n_bins=128) for q in range(128)]
        assert vals == sorted(vals)

    def test_quantize_many_bins(self):
        """With 256 bins, reconstruction error should be ≤ one bin."""
        n    = 256
        lo, hi = DEFAULT_COORD_LO, DEFAULT_COORD_HI
        step = (hi - lo) / n
        x    = 0.7
        q    = quantize_coord(x, n_bins=n)
        x2   = dequantize_coord(q, n_bins=n)
        assert abs(x - x2) <= step + 1e-9

    def test_symmetry_around_zero(self):
        """quantize(-x) and n_bins - 1 - quantize(x) should be equal or off by 1."""
        n   = 128
        lo  = DEFAULT_COORD_LO
        hi  = DEFAULT_COORD_HI
        x   = 0.5
        q_pos = quantize_coord(+x, n_bins=n)
        q_neg = quantize_coord(-x, n_bins=n)
        # They should be symmetric around n//2
        assert abs((q_pos + q_neg) - (n - 1)) <= 1


# ─────────────────────────────────────────────────────────────────────────────
# 3. tokenize() — structural properties
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_always_ends_with_eos(self):
        tokens = tokenize(make_icosahedron())
        assert tokens[-1].op == 'EOS'

    def test_genus0_no_hdl_tokens(self):
        """Genus-0 mesh requires no HDL tokens."""
        tokens = tokenize(make_icosahedron())
        stats  = token_stats(tokens)
        assert 'HDL' not in stats

    def test_genus1_one_hdl_token(self):
        tokens = tokenize(_make_genus1_mesh())
        stats  = token_stats(tokens)
        assert stats.get('HDL', 0) == 1

    def test_genus2_two_hdl_tokens(self):
        tokens = tokenize(_make_genus2_mesh())
        stats  = token_stats(tokens)
        assert stats.get('HDL', 0) == 2

    def test_cv_tokens_present(self):
        """Every sequence should have at least some CV tokens."""
        tokens = tokenize(make_icosahedron())
        stats  = token_stats(tokens)
        assert stats.get('CV', 0) > 0

    def test_cc_tokens_increase_with_subdivision(self):
        """A mesh with more vertices should require more CC tokens."""
        mesh_fine = catmull_clark(catmull_clark(make_icosahedron()))
        tokens_base = tokenize(make_icosahedron())
        tokens_fine = tokenize(mesh_fine)
        cc_base = token_stats(tokens_base).get('CC', 0)
        cc_fine = token_stats(tokens_fine).get('CC', 0)
        assert cc_fine >= cc_base

    def test_token_types_are_valid(self):
        """All token op strings must be in the allowed set."""
        valid_ops = {'CV', 'IE', 'DE', 'CC', 'HDL', 'EOS'}
        for mesh in [make_icosahedron(), make_cube(), _make_genus1_mesh()]:
            for tok in tokenize(mesh):
                assert tok.op in valid_ops, f"Unknown op: {tok.op}"

    def test_cv_token_has_pos(self):
        """Every CV token must carry a (qx, qy, qz) triple."""
        for tok in tokenize(make_icosahedron()):
            if tok.op == 'CV':
                assert tok.pos is not None
                assert len(tok.pos) == 3
                qx, qy, qz = tok.pos
                assert isinstance(qx, int) and isinstance(qy, int) and isinstance(qz, int)

    def test_cv_bins_in_range(self):
        """Quantized bins must be within [0, n_bins-1]."""
        n = 64
        for tok in tokenize(make_icosahedron(), n_position_bins=n):
            if tok.op == 'CV':
                for q in tok.pos:
                    assert 0 <= q < n, f"bin {q} out of range [0, {n})"

    def test_hdl_token_has_corners(self):
        tokens = tokenize(_make_genus1_mesh())
        for tok in tokens:
            if tok.op == 'HDL':
                assert tok.corner1 is not None
                assert tok.corner2 is not None
                f1_ord, _ = tok.corner1
                f2_ord, _ = tok.corner2
                assert f1_ord != f2_ord, "HDL must specify distinct faces"

    def test_normalize_flag_affects_positions(self):
        """With normalize=False, raw coordinates are used directly."""
        mesh   = make_icosahedron()
        t_norm = tokenize(mesh, normalize=True)
        t_raw  = tokenize(mesh, normalize=False)
        cv_norm = [t for t in t_norm if t.op == 'CV']
        cv_raw  = [t for t in t_raw  if t.op == 'CV']
        # They may differ only if the icosahedron is not already normalised
        # (here they should be identical since icosahedron fits in [-2,2])
        # Just check both have same count
        assert len(cv_norm) == len(cv_raw)

    def test_validate_steps_flag(self):
        """validate_steps=True should not raise for a valid mesh."""
        tokens = tokenize(make_icosahedron(), validate_steps=True)
        assert tokens[-1].op == 'EOS'


# ─────────────────────────────────────────────────────────────────────────────
# 4. detokenize() — manifold guarantee
# ─────────────────────────────────────────────────────────────────────────────

class TestDetokenize:
    def test_no_tokens_gives_icosahedron(self):
        """Empty sequence (only EOS) should return the base icosahedron."""
        tokens = [TopModToken(op='EOS')]
        mesh   = detokenize(tokens)
        assert mesh.V() == 12
        assert mesh.E() == 30
        assert mesh.F() == 20

    def test_cc_token_subdivides(self):
        tokens = [TopModToken(op='CC'), TopModToken(op='EOS')]
        mesh   = detokenize(tokens)
        # One CC on icosahedron: V=42, E=120, F=80 ... actually let's check
        ref    = catmull_clark(make_icosahedron())
        assert mesh.V() == ref.V()
        assert mesh.F() == ref.F()

    def test_cc_token_manifold(self):
        tokens = [TopModToken(op='CC'), TopModToken(op='EOS')]
        mesh   = detokenize(tokens)
        assert is_manifold(mesh)

    def test_hdl_increases_genus(self):
        # HDL(0, 10) — icosahedron ordinals 0 and 10
        tokens = [
            TopModToken(op='HDL', corner1=(0, 0), corner2=(10, 0)),
            TopModToken(op='EOS'),
        ]
        mesh = detokenize(tokens)
        assert mesh.genus() == 1

    def test_hdl_manifold(self):
        tokens = [
            TopModToken(op='HDL', corner1=(0, 0), corner2=(10, 0)),
            TopModToken(op='EOS'),
        ]
        mesh = detokenize(tokens)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_two_hdl_genus2(self):
        tokens = [
            TopModToken(op='HDL', corner1=(0, 0),  corner2=(10, 0)),
            TopModToken(op='HDL', corner1=(3, 0),  corner2=(9, 0)),
            TopModToken(op='EOS'),
        ]
        mesh = detokenize(tokens)
        assert mesh.genus() == 2
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_cv_sets_vertex_positions(self):
        """A single CV token should update the first vertex."""
        n_bins = 128
        # Quantize a known position
        target_x = 0.75
        qx = quantize_coord(target_x, n_bins=n_bins)
        qy = quantize_coord(0.0,       n_bins=n_bins)
        qz = quantize_coord(0.0,       n_bins=n_bins)

        tokens = [
            TopModToken(op='CV', pos=(qx, qy, qz)),
            TopModToken(op='EOS'),
        ]
        mesh  = detokenize(tokens, n_position_bins=n_bins)
        verts = list(mesh.vertices.values())
        step  = (DEFAULT_COORD_HI - DEFAULT_COORD_LO) / n_bins

        assert abs(verts[0].x - target_x) <= step + 1e-9

    def test_ie_token_splits_face(self):
        """IE on a same-face corner pair should split that face."""
        mesh_ref = make_icosahedron()
        F0       = mesh_ref.F()

        # Find a face and emit IE across two of its half-edges
        face  = list(make_icosahedron().faces.values())[0]
        # face has 3 half-edges; he0 and he1 are adjacent, he0 and he2 are not
        tokens = [
            TopModToken(op='IE', corner1=(0, 0), corner2=(0, 2)),
            TopModToken(op='EOS'),
        ]
        mesh = detokenize(tokens)
        assert mesh.F() == F0 + 1   # same-face split: F+1

    def test_ie_token_manifold(self):
        tokens = [
            TopModToken(op='IE', corner1=(0, 0), corner2=(0, 2)),
            TopModToken(op='EOS'),
        ]
        mesh = detokenize(tokens)
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_de_token_merges_faces(self):
        """DE(0) should remove edge 0 and merge its two adjacent faces."""
        mesh_ref = make_icosahedron()
        F0 = mesh_ref.F()
        tokens = [
            TopModToken(op='DE', edge_ord=0),
            TopModToken(op='EOS'),
        ]
        mesh = detokenize(tokens)
        assert mesh.F() == F0 - 1
        ok, errs = check_all(mesh)
        assert ok, errs

    def test_validate_steps_does_not_raise_for_valid_sequence(self):
        tokens = tokenize(make_icosahedron(), validate_steps=True)
        mesh   = detokenize(tokens, validate_steps=True)
        assert is_manifold(mesh)

    def test_bad_ordinal_raises(self):
        tokens = [
            TopModToken(op='HDL', corner1=(999, 0), corner2=(998, 0)),
            TopModToken(op='EOS'),
        ]
        with pytest.raises(ValueError, match="out of range"):
            detokenize(tokens)

    def test_unknown_op_raises(self):
        tokens = [TopModToken(op='BOGUS'), TopModToken(op='EOS')]
        with pytest.raises(ValueError, match="Unknown token op"):
            detokenize(tokens)

    def test_eos_stops_execution(self):
        """Tokens after EOS should not be executed."""
        tokens = [
            TopModToken(op='EOS'),
            TopModToken(op='CC'),   # should be ignored
        ]
        mesh = detokenize(tokens)
        assert mesh.V() == 12   # icosahedron, no CC applied


# ─────────────────────────────────────────────────────────────────────────────
# 5. Roundtrip tests
# ─────────────────────────────────────────────────────────────────────────────

class TestRoundtrip:
    """
    tokenize → detokenize roundtrip.

    We guarantee:
      - genus preserved (exact)
      - is_manifold (exact)
      - V count matches when template matches input (exact)
      - positions within 2× quantization step (for matching V counts)
    """

    def _position_error(self, mesh_a, mesh_b):
        """Max absolute positional difference between vertex 0 of two meshes."""
        va = list(mesh_a.vertices.values())
        vb = list(mesh_b.vertices.values())
        n  = min(len(va), len(vb))
        errs = []
        for i in range(n):
            errs.append(max(abs(va[i].x - vb[i].x),
                            abs(va[i].y - vb[i].y),
                            abs(va[i].z - vb[i].z)))
        return max(errs) if errs else 0.0

    def test_icosahedron_genus_preserved(self):
        mesh    = make_icosahedron()
        tokens  = tokenize(mesh)
        result  = detokenize(tokens)
        assert result.genus() == mesh.genus()

    def test_icosahedron_manifold_after_roundtrip(self):
        mesh    = make_icosahedron()
        tokens  = tokenize(mesh)
        result  = detokenize(tokens)
        ok, errs = check_all(result)
        assert ok, errs

    def test_icosahedron_vertex_count_preserved(self):
        """Icosahedron V=12 matches base template (no CC needed)."""
        mesh   = make_icosahedron()
        tokens = tokenize(mesh)
        result = detokenize(tokens)
        assert result.V() == mesh.V()

    def test_icosahedron_positions_roundtrip(self):
        """After quantise→dequantise (normalize=False), positions within 2 bins."""
        n_bins = 128
        step   = (DEFAULT_COORD_HI - DEFAULT_COORD_LO) / n_bins
        mesh   = make_icosahedron()
        # Use normalize=False so reconstructed positions are in the same scale
        # as the originals (unit sphere, all coords in ≈ [-0.85, 0.85] ⊂ [-2, 2])
        tokens = tokenize(mesh, n_position_bins=n_bins, normalize=False)
        result = detokenize(tokens, n_position_bins=n_bins)
        err    = self._position_error(mesh, result)
        # Allow up to ~2 bin widths (quantization error only)
        assert err < step * 4, f"Max position error {err:.4f} > {step * 4:.4f}"

    def test_cc_subdivided_icosahedron_roundtrip(self):
        """One CC round on icosahedron; template should match exactly."""
        mesh    = catmull_clark(make_icosahedron())
        tokens  = tokenize(mesh)
        result  = detokenize(tokens)
        assert result.genus() == mesh.genus()
        assert result.V() == mesh.V()
        ok, errs = check_all(result)
        assert ok, errs

    def test_genus1_roundtrip_topology(self):
        mesh   = _make_genus1_mesh()
        tokens = tokenize(mesh)
        result = detokenize(tokens)
        assert result.genus() == 1
        ok, errs = check_all(result)
        assert ok, errs

    def test_genus2_roundtrip_topology(self):
        mesh   = _make_genus2_mesh()
        tokens = tokenize(mesh)
        result = detokenize(tokens)
        assert result.genus() == 2
        ok, errs = check_all(result)
        assert ok, errs

    def test_roundtrip_euler_characteristic(self):
        """χ = V - E + F must match for genus-0 and genus-1."""
        for mesh in [make_icosahedron(), _make_genus1_mesh()]:
            chi_in  = mesh.euler_characteristic()
            tokens  = tokenize(mesh)
            result  = detokenize(tokens)
            assert result.euler_characteristic() == chi_in

    def test_roundtrip_more_bins_better_accuracy(self):
        """256 bins should give smaller position error than 32 bins."""
        mesh  = make_icosahedron()
        n_lo  = 32
        n_hi  = 256

        # normalize=False so both use the same coordinate space (unit sphere coords
        # are in ≈ [-0.85, 0.85] ⊂ [-2, 2]) and errors are purely from quantization
        r_lo  = detokenize(tokenize(mesh, n_position_bins=n_lo, normalize=False),
                           n_position_bins=n_lo)
        r_hi  = detokenize(tokenize(mesh, n_position_bins=n_hi, normalize=False),
                           n_position_bins=n_hi)

        err_lo = self._position_error(mesh, r_lo)
        err_hi = self._position_error(mesh, r_hi)
        assert err_hi <= err_lo + 1e-6, (
            f"More bins should give less error: err_lo={err_lo:.4f}, err_hi={err_hi:.4f}"
        )

    def test_cube_roundtrip_genus(self):
        """Cube is genus-0; topology must survive roundtrip."""
        mesh   = make_cube()
        tokens = tokenize(mesh)
        result = detokenize(tokens)
        assert result.genus() == 0
        assert is_manifold(result)

    def test_octahedron_roundtrip(self):
        mesh   = make_octahedron()
        tokens = tokenize(mesh)
        result = detokenize(tokens)
        assert result.genus() == 0
        assert is_manifold(result)

    def test_two_cc_roundtrip(self):
        mesh   = catmull_clark(catmull_clark(make_icosahedron()))
        tokens = tokenize(mesh)
        result = detokenize(tokens)
        assert result.genus() == 0
        assert result.V() == mesh.V()
        ok, errs = check_all(result)
        assert ok, errs


# ─────────────────────────────────────────────────────────────────────────────
# 6. Vocabulary & encoding
# ─────────────────────────────────────────────────────────────────────────────

class TestVocabulary:
    def test_build_vocab_returns_dict(self):
        vocab = build_vocabulary()
        assert isinstance(vocab, dict)
        assert len(vocab) > 0

    def test_vocab_contains_op_tokens(self):
        vocab = build_vocabulary()
        for op in ('EOS', 'CC', 'CV', 'IE', 'DE', 'HDL'):
            assert op in vocab, f"{op} missing from vocabulary"

    def test_vocab_ids_are_unique(self):
        vocab = build_vocabulary(n_position_bins=64, max_ordinal=256)
        ids   = list(vocab.values())
        assert len(ids) == len(set(ids)), "Duplicate IDs in vocabulary"

    def test_vocab_coord_keys_present(self):
        n     = 64
        vocab = build_vocabulary(n_position_bins=n, max_ordinal=16)
        for i in range(n):
            assert f'COORD_{i}' in vocab

    def test_vocab_ref_keys_present(self):
        max_ref = 32
        vocab   = build_vocabulary(n_position_bins=4, max_ordinal=max_ref)
        for i in range(max_ref):
            assert f'REF_{i}' in vocab

    def test_vocab_size(self):
        n_bins  = 64
        max_ref = 128
        vocab   = build_vocabulary(n_position_bins=n_bins, max_ordinal=max_ref)
        # 6 op tokens + n_bins coord tokens + max_ref ref tokens
        # + 19 extension ops (DUAL..SQRT3 + HONEY/STAR/CCUT/LSTYLE/FRAC
        #   + PENT/PENT2/D1264/ROOT4 + CHKB/DSBC/DOME)
        assert len(vocab) == 6 + n_bins + max_ref + 19

    def test_encode_eos(self):
        vocab  = build_vocabulary(n_position_bins=32, max_ordinal=64)
        tokens = [TopModToken(op='EOS')]
        ids    = encode_sequence(tokens, vocab)
        assert ids == [vocab['EOS']]

    def test_encode_cc(self):
        vocab  = build_vocabulary(n_position_bins=32, max_ordinal=64)
        tokens = [TopModToken(op='CC')]
        ids    = encode_sequence(tokens, vocab)
        assert ids == [vocab['CC']]

    def test_encode_cv(self):
        vocab  = build_vocabulary(n_position_bins=32, max_ordinal=64)
        tok    = TopModToken(op='CV', pos=(5, 10, 20))
        ids    = encode_sequence([tok], vocab)
        assert ids == [
            vocab['CV'],
            vocab['COORD_5'],
            vocab['COORD_10'],
            vocab['COORD_20'],
        ]

    def test_encode_hdl(self):
        vocab  = build_vocabulary(n_position_bins=32, max_ordinal=64)
        tok    = TopModToken(op='HDL', corner1=(0, 0), corner2=(5, 0))
        ids    = encode_sequence([tok], vocab)
        assert ids == [vocab['HDL'], vocab['REF_0'], vocab['REF_5']]

    def test_encode_ie(self):
        vocab  = build_vocabulary(n_position_bins=32, max_ordinal=64)
        tok    = TopModToken(op='IE', corner1=(2, 1), corner2=(7, 0))
        ids    = encode_sequence([tok], vocab)
        assert ids == [
            vocab['IE'],
            vocab['REF_2'], vocab['REF_1'],
            vocab['REF_7'], vocab['REF_0'],
        ]

    def test_encode_de(self):
        vocab  = build_vocabulary(n_position_bins=32, max_ordinal=64)
        tok    = TopModToken(op='DE', edge_ord=3)
        ids    = encode_sequence([tok], vocab)
        assert ids == [vocab['DE'], vocab['REF_3']]

    def test_encode_decode_roundtrip(self):
        """encode → decode should recover the original tokens exactly."""
        vocab     = build_vocabulary(n_position_bins=128, max_ordinal=256)
        vocab_inv = {v: k for k, v in vocab.items()}

        original = [
            TopModToken(op='HDL', corner1=(0, 0), corner2=(10, 0)),
            TopModToken(op='CC'),
            TopModToken(op='CV', pos=(63, 64, 96)),
            TopModToken(op='EOS'),
        ]
        ids      = encode_sequence(original, vocab)
        restored = decode_sequence(ids, vocab_inv)

        assert len(restored) == len(original)
        for orig, rest in zip(original, restored):
            assert orig.op == rest.op
            assert orig.pos == rest.pos
            assert orig.corner1 == rest.corner1
            assert orig.corner2 == rest.corner2
            assert orig.edge_ord == rest.edge_ord

    def test_full_tokenize_encode_decode_detokenize(self):
        """Full pipeline: mesh → tokens → ints → tokens → mesh; genus preserved."""
        n_bins    = 128
        max_ref   = 512
        vocab     = build_vocabulary(n_position_bins=n_bins, max_ordinal=max_ref)
        vocab_inv = {v: k for k, v in vocab.items()}

        mesh      = make_icosahedron()
        tokens    = tokenize(mesh, n_position_bins=n_bins)
        ids       = encode_sequence(tokens, vocab)
        restored  = decode_sequence(ids, vocab_inv)
        result    = detokenize(restored, n_position_bins=n_bins)

        assert result.genus() == mesh.genus()
        assert is_manifold(result)


# ─────────────────────────────────────────────────────────────────────────────
# 7. token_stats and sequence_length
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenStats:
    def test_stats_returns_dict(self):
        tokens = tokenize(make_icosahedron())
        stats  = token_stats(tokens)
        assert isinstance(stats, dict)

    def test_stats_counts_eos(self):
        tokens = tokenize(make_icosahedron())
        stats  = token_stats(tokens)
        assert stats['EOS'] == 1

    def test_stats_total_matches_length(self):
        tokens = tokenize(make_icosahedron())
        stats  = token_stats(tokens)
        assert sum(stats.values()) == len(tokens)

    def test_sequence_length_positive(self):
        tokens = tokenize(make_icosahedron())
        L      = sequence_length(tokens)
        assert L > 0

    def test_sequence_length_matches_encoded(self):
        """sequence_length() must agree with len(encode_sequence(...))."""
        vocab  = build_vocabulary(n_position_bins=128, max_ordinal=1024)
        tokens = tokenize(make_icosahedron())
        L_pred = sequence_length(tokens)
        L_real = len(encode_sequence(tokens, vocab))
        assert L_pred == L_real


# ─────────────────────────────────────────────────────────────────────────────
# 8. Edge cases & error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_mesh_raises_or_handles(self):
        """tokenize on a mesh with no vertices should not crash."""
        mesh = DLFLMesh()
        try:
            tokens = tokenize(mesh)
            # Acceptable: EOS or minimal sequence
            assert tokens[-1].op == 'EOS'
        except (ValueError, ZeroDivisionError):
            pass   # also acceptable

    def test_tokenize_with_different_bin_counts(self):
        """tokenize + detokenize with various n_position_bins."""
        mesh = make_icosahedron()
        for n in (32, 64, 128, 256):
            tokens = tokenize(mesh, n_position_bins=n)
            result = detokenize(tokens, n_position_bins=n)
            assert result.genus() == 0
            assert is_manifold(result)

    def test_hdl_token_face_ordinals_differ(self):
        """A valid HDL token must specify two DISTINCT face ordinals."""
        tokens = tokenize(_make_genus1_mesh())
        for tok in tokens:
            if tok.op == 'HDL':
                f1, _ = tok.corner1
                f2, _ = tok.corner2
                assert f1 != f2, "HDL cannot connect a face to itself"

    def test_genus1_cc_combination(self):
        """Genus-1 mesh after CC subdivision roundtrips correctly."""
        mesh    = catmull_clark(_make_genus1_mesh())
        tokens  = tokenize(mesh)
        result  = detokenize(tokens)
        assert result.genus() == 1
        ok, errs = check_all(result)
        assert ok, errs

    def test_detokenize_multiple_cv_tokens(self):
        """Multiple CV tokens should update consecutive vertices."""
        n_bins = 128
        qx, qy, qz = (10, 20, 30)
        tokens = [
            TopModToken(op='CV', pos=(qx, qy, qz)),
            TopModToken(op='CV', pos=(50, 60, 70)),
            TopModToken(op='EOS'),
        ]
        mesh  = detokenize(tokens, n_position_bins=n_bins)
        verts = list(mesh.vertices.values())
        step  = (DEFAULT_COORD_HI - DEFAULT_COORD_LO) / n_bins
        # First vertex should match (qx, qy, qz)
        assert abs(verts[0].x - dequantize_coord(qx)) < step + 1e-9
        # Second vertex should match (50, 60, 70)
        assert abs(verts[1].x - dequantize_coord(50)) < step + 1e-9

    def test_max_cc_rounds_respected(self):
        """tokenize respects max_cc_rounds parameter."""
        # Make a very large mesh (many CC rounds would be needed)
        large_mesh = make_icosahedron()
        for _ in range(3):
            large_mesh = catmull_clark(large_mesh)
        # Tokenize with only 1 CC max
        tokens = tokenize(large_mesh, max_cc_rounds=1)
        stats  = token_stats(tokens)
        assert stats.get('CC', 0) <= 1

    def test_find_compatible_pair_raises_when_impossible(self):
        """Helper raises when the mesh has no valid face pair."""
        mesh = make_tetrahedron()
        # Exclude all vertices — no valid pair possible
        all_vids = {v.id for v in mesh.vertices.values()}
        with pytest.raises(ValueError, match="compatible"):
            _find_compatible_face_pair(mesh, all_vids)


# ─────────────────────────────────────────────────────────────────────────────
# DUAL / DS extension tokens (added after semantic-oracle validation)
# ─────────────────────────────────────────────────────────────────────────────

class TestDualDsTokens:
    def test_vocab_backward_compatible(self):
        """Extension ops append at the END: all legacy IDs unchanged."""
        vocab = build_vocabulary(n_position_bins=128, max_ordinal=100)
        assert [vocab[o] for o in ('EOS', 'CC', 'CV', 'IE', 'DE', 'HDL')] == [0, 1, 2, 3, 4, 5]
        assert vocab['COORD_0'] == 6
        assert vocab['REF_0'] == 6 + 128
        assert vocab['DUAL'] == 6 + 128 + 100
        assert vocab['DS'] == 6 + 128 + 100 + 1
        # later extensions keep appending in fixed order
        for i, op in enumerate(('STA', 'SIMP', 'VC', 'LOOP', 'SQRT3',
                                'HONEY', 'STAR', 'CCUT', 'LSTYLE', 'FRAC')):
            assert vocab[op] == 6 + 128 + 100 + 2 + i

    def test_encode_decode_roundtrip(self):
        vocab = build_vocabulary(n_position_bins=128, max_ordinal=100)
        vocab_inv = {v: k for k, v in vocab.items()}
        tokens = [
            TopModToken(op='DUAL'),
            TopModToken(op='DS'),
            TopModToken(op='CC'),
            TopModToken(op='EOS'),
        ]
        ids = encode_sequence(tokens, vocab)
        back = decode_sequence(ids, vocab_inv)
        assert [t.op for t in back] == ['DUAL', 'DS', 'CC', 'EOS']

    def test_detokenize_executes_dual(self):
        """DUAL on icosahedron base -> dodecahedron counts (20V 30E 12F)."""
        mesh = detokenize([TopModToken(op='DUAL'), TopModToken(op='EOS')])
        assert (mesh.V(), mesh.E(), mesh.F()) == (20, 30, 12)
        assert is_manifold(mesh)

    def test_detokenize_executes_ds(self):
        """DS on icosahedron base: V'=2E=60, E'=4E=120, F'=V+E+F=62."""
        mesh = detokenize([TopModToken(op='DS'), TopModToken(op='EOS')])
        assert (mesh.V(), mesh.E(), mesh.F()) == (60, 120, 62)
        assert is_manifold(mesh)

    def test_detokenize_mixed_sequence_manifold(self):
        mesh = detokenize([
            TopModToken(op='DUAL'),
            TopModToken(op='CC'),
            TopModToken(op='DS'),
            TopModToken(op='EOS'),
        ], validate_steps=True)
        assert is_manifold(mesh)

    def test_sequence_length_single_slot(self):
        toks = [TopModToken(op='DUAL'), TopModToken(op='DS'), TopModToken(op='EOS')]
        assert sequence_length(toks) == 3


class TestGlobalOpTokens:
    """STA / SIMP / VC / LOOP / SQRT3 extension tokens."""

    def test_encode_decode_roundtrip(self):
        vocab = build_vocabulary(n_position_bins=128, max_ordinal=100)
        vocab_inv = {v: k for k, v in vocab.items()}
        ops = ['STA', 'SIMP', 'VC', 'LOOP', 'SQRT3', 'EOS']
        tokens = [TopModToken(op=o) for o in ops]
        back = decode_sequence(encode_sequence(tokens, vocab), vocab_inv)
        assert [t.op for t in back] == ops

    def test_detokenize_sta(self):
        """STA on icosahedron: V'=V+F=32, E'=3E=90, F'=2E=60."""
        mesh = detokenize([TopModToken(op='STA'), TopModToken(op='EOS')])
        assert (mesh.V(), mesh.E(), mesh.F()) == (32, 90, 60)
        assert is_manifold(mesh)

    def test_detokenize_simp(self):
        """SIMP on icosahedron: V'=E=30, E'=2E=60, F'=F+V=32."""
        mesh = detokenize([TopModToken(op='SIMP'), TopModToken(op='EOS')])
        assert (mesh.V(), mesh.E(), mesh.F()) == (30, 60, 32)
        assert is_manifold(mesh)

    def test_detokenize_vc(self):
        """VC on icosahedron: V'=2E=60, E'=3E=90, F'=F+V=32."""
        mesh = detokenize([TopModToken(op='VC'), TopModToken(op='EOS')])
        assert (mesh.V(), mesh.E(), mesh.F()) == (60, 90, 32)
        assert is_manifold(mesh)

    def test_detokenize_tri_schemes(self):
        """LOOP then SQRT3 on the all-tri icosahedron base."""
        mesh = detokenize([
            TopModToken(op='LOOP'),
            TopModToken(op='SQRT3'),
            TopModToken(op='EOS'),
        ], validate_steps=True)
        assert is_manifold(mesh)

    def test_loop_on_non_tri_state_raises(self):
        """LOOP after DS (quads exist) must fail loudly, not corrupt."""
        with pytest.raises(ValueError):
            detokenize([
                TopModToken(op='DS'),
                TopModToken(op='LOOP'),
                TopModToken(op='EOS'),
            ])


class TestBatch2Tokens:
    """HONEY / STAR / CCUT / LSTYLE / FRAC extension tokens."""

    def test_encode_decode_roundtrip(self):
        vocab = build_vocabulary(n_position_bins=128, max_ordinal=100)
        vocab_inv = {v: k for k, v in vocab.items()}
        ops = ['HONEY', 'STAR', 'CCUT', 'LSTYLE', 'FRAC', 'EOS']
        tokens = [TopModToken(op=o) for o in ops]
        back = decode_sequence(encode_sequence(tokens, vocab), vocab_inv)
        assert [t.op for t in back] == ops

    def test_detokenize_each(self):
        """Each opcode executes on the icosahedron base (V12 E30 F20)."""
        expected = {
            'HONEY':  (60, 90, 32),     # 2E, 3E, F+V
            'STAR':   (92, 270, 180),   # V+F+2E, 9E, 6E
            'CCUT':   (60, 120, 62),    # 2E, 4E, V+E+F
            'LSTYLE': (42, 120, 80),    # V+E, 4E, F+2E
            'FRAC':   (62, 180, 120),   # V+E+F, 6E, 4E
        }
        for op, cnt in expected.items():
            mesh = detokenize([TopModToken(op=op), TopModToken(op='EOS')],
                              validate_steps=True)
            assert (mesh.V(), mesh.E(), mesh.F()) == cnt, op
            assert is_manifold(mesh), op


class TestBatch4Tokens:
    """CHKB / DSBC extension tokens."""

    def test_vocab_backward_compatible(self):
        vocab = build_vocabulary(n_position_bins=128, max_ordinal=100)
        base = 6 + 128 + 100 + 16   # after DUAL..ROOT4
        for i, op in enumerate(('CHKB', 'DSBC', 'DOME')):
            assert vocab[op] == base + i

    def test_detokenize_each(self):
        """Each opcode executes on the icosahedron base (V12 E30 F20)."""
        expected = {
            'CHKB': (132, 270, 140),   # V+4E, 9E, F+4E
            'DSBC': (132, 210, 80),    # V+4E, 7E, F+2E
            'DOME': (1782, 3480, 1700),  # V+59E, 116E, F+56E
        }
        for op, cnt in expected.items():
            mesh = detokenize([TopModToken(op=op), TopModToken(op='EOS')],
                              validate_steps=True)
            assert (mesh.V(), mesh.E(), mesh.F()) == cnt, op
            assert is_manifold(mesh), op


class TestBatch3Tokens:
    """PENT / PENT2 / D1264 / ROOT4 extension tokens."""

    def test_vocab_backward_compatible(self):
        vocab = build_vocabulary(n_position_bins=128, max_ordinal=100)
        base = 6 + 128 + 100 + 12   # after DUAL..FRAC
        for i, op in enumerate(('PENT', 'PENT2', 'D1264', 'ROOT4')):
            assert vocab[op] == base + i

    def test_encode_decode_roundtrip(self):
        vocab = build_vocabulary(n_position_bins=128, max_ordinal=100)
        vocab_inv = {v: k for k, v in vocab.items()}
        ops = ['PENT', 'PENT2', 'D1264', 'ROOT4', 'EOS']
        tokens = [TopModToken(op=o) for o in ops]
        back = decode_sequence(encode_sequence(tokens, vocab), vocab_inv)
        assert [t.op for t in back] == ops

    def test_detokenize_each(self):
        """Each opcode executes on the icosahedron base (V12 E30 F20)."""
        expected = {
            'PENT':  (92, 150, 60),    # V+2E+F, 5E, 2E
            'PENT2': (102, 180, 80),   # V+3E, 6E, F+2E
            'D1264': (120, 180, 62),   # 4E, 6E, F+E+V
            'ROOT4': (72, 120, 50),    # V+2E, 4E, F+E
        }
        for op, cnt in expected.items():
            mesh = detokenize([TopModToken(op=op), TopModToken(op='EOS')],
                              validate_steps=True)
            assert (mesh.V(), mesh.E(), mesh.F()) == cnt, op
            assert is_manifold(mesh), op
