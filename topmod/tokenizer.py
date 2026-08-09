"""
TopMod Tokenizer — reverse decomposition of meshes into operator sequences.

Implements the 'topology-first' tokenization strategy proposed in
docs/paper_integration.md (Priority 1: MeshGPT Integration).

Core idea
---------
Every valid orientable 2-manifold mesh can be expressed as a finite sequence
of TopMod operators applied to a fixed base primitive (icosahedron).
This sequence is the "token representation" of the mesh.

Token vocabulary
----------------
  HDL  (face_ord1, face_ord2)    — add_handle macro (increases genus)
  IE   (f_ord, he_pos, f_ord2, he_pos2) — single insert_edge
  DE   (edge_ord,)               — single delete_edge
  CC   ()                        — one Catmull-Clark subdivision round
  DUAL ()                        — combinatorial dual (V'=F, F'=V, E'=E)
  DS   ()                        — one Doo-Sabin subdivision round
  STA  ()                        — stellate all faces (V'=V+F, E'=3E, F'=2E)
  SIMP ()                        — mid-edge subdivision (V'=E, E'=2E, F'=F+V)
  VC   ()                        — vertex cutting (V'=2E, E'=3E, F'=F+V)
  LOOP ()                        — Loop subdivision (tri only; V'=V+E)
  SQRT3()                        — sqrt(3) subdivision (tri only; V'=V+F)
  HONEY()                        — honeycomb = dual∘stellate_all (V'=2E)
  STAR ()                        — star = stellate_all² (V'=V+F+2E)
  CCUT ()                        — corner cutting (DS topology, α default)
  LSTYLE()                       — loop-style split (V'=V+E, F'=F+2E)
  FRAC ()                        — fractal = loop_style + apex (V'=V+E+F)
  PENT ()                        — pentagonal (V'=V+2E+F, all pentagons)
  PENT2()                        — pentagonal variant 2 (V'=V+3E)
  D1264()                        — dual 12.6.4 (V'=4E, F'=F+E+V)
  ROOT4()                        — root-4 (V'=V+2E, F'=F+E)
  CHKB ()                        — checkerboard (V'=V+4E, F'=F+4E)
  DSBC ()                        — Doo-Sabin BC-new (V'=V+4E, F'=F+2E)
  DOME ()                        — dome (V'=V+59E, E'=116E, F'=F+56E)
  CRUST()                        — crust/shell (V'=2V, E'=2E, F'=2F, 2 comps;
                                   punch holes with HDL on mirror pairs i↔F+i)
  CV   (qx, qy, qz)             — set next vertex position (quantized)
  EOS  ()                        — end-of-sequence

Token parameters use *ordinal indices* into the mesh's face/edge/vertex lists
(insertion-order, consistent between tokenize and detokenize because both
always start from the same deterministic base state).

Guarantee
---------
After executing any valid token sequence, is_manifold(mesh) is True.

Public API
----------
TopModToken                     — dataclass
quantize_coord / dequantize_coord
tokenize(mesh, **kwargs)        → List[TopModToken]
detokenize(tokens, **kwargs)    → DLFLMesh
build_vocabulary(**kwargs)      → dict[str, int]
encode_sequence(tokens, vocab)  → List[int]
decode_sequence(ids, vocab_inv) → List[TopModToken]
token_stats(tokens)             → dict
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import sys, os
_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, os.path.join(_HERE, ".."))

from .dlfl import DLFLMesh, Face, HalfEdge, Vertex
from .operators import insert_edge, delete_edge
from .high_level_ops import add_handle
from .subdivision import catmull_clark
from .remeshing import (dual, doo_sabin, simplest_subdivide,
                        vertex_cutting, loop_subdivide, sqrt3_subdivide,
                        honeycomb_subdivide, star_subdivide, corner_cutting,
                        loop_style_subdivide, fractal_subdivide,
                        pentagonal_subdivide, pentagonal2_subdivide,
                        dual1264_subdivide, root4_subdivide,
                        checkerboard_remesh, ds_bc_new_subdivide,
                        dome_subdivide, create_crust)
from .high_level_ops import stellate_all
from .primitives import make_icosahedron
from .validate import is_manifold, check_all


# ═══════════════════════════════════════════════════════════════════════════════
# Token dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TopModToken:
    """
    One token in a TopMod operator sequence.

    Parameters
    ----------
    op : str
        Token type: 'CV' | 'IE' | 'DE' | 'CC' | 'HDL' | 'EOS'

    pos : (qx, qy, qz) | None
        CV only: quantized coordinate triple (int bins).

    corner1, corner2 : (face_ordinal, he_pos) | None
        IE / HDL: corners specified as (face index in insertion-order face list,
        halfedge position within that face's boundary loop).
        For HDL, he_pos is always 0 (any halfedge of the face suffices).

    edge_ord : int | None
        DE only: index of the edge in insertion-order edge list.
    """
    op: str   # 'CV' | 'IE' | 'DE' | 'CC' | 'DUAL' | 'DS' | 'HDL' | 'EOS'

    pos:      Optional[Tuple[int, int, int]] = None   # CV
    corner1:  Optional[Tuple[int, int]]      = None   # IE / HDL
    corner2:  Optional[Tuple[int, int]]      = None   # IE / HDL
    edge_ord: Optional[int]                  = None   # DE

    def __repr__(self) -> str:
        if self.op == 'CV':
            return f"CV({self.pos[0]},{self.pos[1]},{self.pos[2]})"
        if self.op in ('IE', 'HDL'):
            c1 = f"f{self.corner1[0]}:h{self.corner1[1]}" if self.corner1 else "?"
            c2 = f"f{self.corner2[0]}:h{self.corner2[1]}" if self.corner2 else "?"
            return f"{self.op}({c1}, {c2})"
        if self.op == 'DE':
            return f"DE(e{self.edge_ord})"
        return self.op


# ═══════════════════════════════════════════════════════════════════════════════
# Coordinate quantization
# ═══════════════════════════════════════════════════════════════════════════════

DEFAULT_COORD_LO: float = -2.0
DEFAULT_COORD_HI: float = +2.0


def quantize_coord(
    x:      float,
    lo:     float = DEFAULT_COORD_LO,
    hi:     float = DEFAULT_COORD_HI,
    n_bins: int   = 128,
) -> int:
    """
    Map a float *x* in [lo, hi] to an integer bin in [0, n_bins − 1].

    Values outside [lo, hi] are clamped.
    """
    t = (x - lo) / (hi - lo)
    return int(max(0, min(n_bins - 1, int(t * n_bins))))


def dequantize_coord(
    q:      int,
    lo:     float = DEFAULT_COORD_LO,
    hi:     float = DEFAULT_COORD_HI,
    n_bins: int   = 128,
) -> float:
    """
    Map a quantized bin index *q* back to the centre of its bin in [lo, hi].
    """
    return lo + (q + 0.5) / n_bins * (hi - lo)


# ═══════════════════════════════════════════════════════════════════════════════
# Internal mesh helpers (ordinal-based)
# ═══════════════════════════════════════════════════════════════════════════════

def _face_ordinal(mesh: DLFLMesh, target: Face) -> int:
    """Return the 0-based position of *target* in the mesh's insertion-order face list."""
    for i, f in enumerate(mesh.faces.values()):
        if f is target:
            return i
    raise ValueError(f"Face {target.id} not in mesh (F={mesh.F()})")


def _edge_ordinal(mesh: DLFLMesh, target) -> int:
    """Return the 0-based position of *target* in the mesh's edge list."""
    for i, e in enumerate(mesh.edges.values()):
        if e is target:
            return i
    raise ValueError(f"Edge {target.id} not in mesh (E={mesh.E()})")


def _he_pos_in_face(face: Face, he: HalfEdge) -> int:
    """Return the position of *he* in face.halfedges() (starting from face.he)."""
    for i, h in enumerate(face.halfedges()):
        if h is he:
            return i
    raise ValueError(f"HalfEdge {he.id} not in face {face.id}")


def _find_compatible_face_pair(
    mesh:         DLFLMesh,
    exclude_vids: Set[int],
    min_degree:   int = 3,
) -> Tuple[Face, Face]:
    """
    Find two faces in *mesh* that:

    1. Both have degree ≥ min_degree.
    2. Neither touches any vertex in *exclude_vids*.
    3. They share no vertex with each other.

    Raises ValueError if no such pair exists.
    """
    candidates: List[Tuple[Face, Set[int]]] = []

    for f in mesh.faces.values():
        if f.degree() < min_degree:
            continue
        vids = {v.id for v in f.vertices()}
        if vids & exclude_vids:
            continue
        candidates.append((f, vids))

    for i, (f1, vids1) in enumerate(candidates):
        for f2, vids2 in candidates[i + 1 :]:
            if not (vids1 & vids2):
                return f1, f2

    raise ValueError(
        f"No compatible face pair found "
        f"(candidates={len(candidates)}, excluded={len(exclude_vids)}). "
        "Try using a base mesh with more faces."
    )


def _mesh_bbox(mesh: DLFLMesh) -> Tuple[float, float]:
    """Return (lo, hi) scalar bounding box (uniform over all axes)."""
    coords = []
    for v in mesh.vertices.values():
        coords.extend([v.x, v.y, v.z])
    if not coords:
        return DEFAULT_COORD_LO, DEFAULT_COORD_HI
    return min(coords), max(coords)


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenize — mesh → token sequence
# ═══════════════════════════════════════════════════════════════════════════════

def tokenize(
    mesh:             DLFLMesh,
    n_position_bins:  int   = 128,
    max_cc_rounds:    int   = 5,
    coord_lo:         float = DEFAULT_COORD_LO,
    coord_hi:         float = DEFAULT_COORD_HI,
    normalize:        bool  = True,
    validate_steps:   bool  = False,
) -> List[TopModToken]:
    """
    Decompose *mesh* into a TopMod token sequence (template-based).

    Strategy
    --------
    1. Compute genus *g* of the input mesh.
    2. Start from make_icosahedron() (genus-0 base — implicit in detokenize).
    3. Emit HDL tokens to add *g* topological handles (genus +1 each).
    4. Emit CC tokens to subdivide until the working mesh has ≥ half the
       target vertex count (or *max_cc_rounds* rounds).
    5. Emit CV tokens for each vertex position of the working mesh
       (mapped from the input mesh by ordinal, or from the working mesh
       if counts differ).
    6. Emit EOS.

    Parameters
    ----------
    mesh            : Input DLFLMesh to tokenize.
    n_position_bins : Number of quantization bins per axis (default 128).
    max_cc_rounds   : Maximum Catmull-Clark rounds to emit (default 5).
    coord_lo / hi   : Coordinate range for quantization.
    normalize       : If True, automatically fit mesh positions into
                      [coord_lo+margin, coord_hi-margin].
    validate_steps  : If True, run is_manifold() after each structural op.

    Returns
    -------
    List[TopModToken]  — sequence ending with EOS.
    """
    tokens: List[TopModToken] = []

    # ── Coordinate range ──────────────────────────────────────────────
    lo, hi = coord_lo, coord_hi
    if normalize:
        mn, mx = _mesh_bbox(mesh)
        extent = max(mx - mn, 1e-6)
        centre = (mn + mx) / 2.0
        # Scale so the mesh spans 0.8 × (hi - lo)
        scale  = 0.8 * (coord_hi - coord_lo) / extent
        # Build normalised position list
        target_positions = [
            (
                (v.x - centre) * scale,
                (v.y - centre) * scale,
                (v.z - centre) * scale,
            )
            for v in mesh.vertices.values()
        ]
    else:
        target_positions = [(v.x, v.y, v.z) for v in mesh.vertices.values()]

    # ── Topology analysis ─────────────────────────────────────────────
    g        = mesh.genus()
    V_target = mesh.V()

    # ── Build working mesh ────────────────────────────────────────────
    working_mesh = make_icosahedron()

    # ── HDL tokens for topological handles ───────────────────────────
    excluded_vids: Set[int] = set()

    for handle_idx in range(g):
        f1, f2 = _find_compatible_face_pair(working_mesh, excluded_vids)

        f1_ord = _face_ordinal(working_mesh, f1)
        f2_ord = _face_ordinal(working_mesh, f2)

        # Exclude vertices of both faces from future handle searches
        excluded_vids |= {v.id for v in f1.vertices()}
        excluded_vids |= {v.id for v in f2.vertices()}

        tokens.append(TopModToken(op='HDL',
                                   corner1=(f1_ord, 0),
                                   corner2=(f2_ord, 0)))
        add_handle(working_mesh, f1, f2)

        if validate_steps and not is_manifold(working_mesh):
            raise RuntimeError(f"Mesh became non-manifold after handle {handle_idx + 1}")

    # ── CC tokens for subdivision ─────────────────────────────────────
    n_cc = 0
    while working_mesh.V() < V_target // 2 and n_cc < max_cc_rounds:
        tokens.append(TopModToken(op='CC'))
        working_mesh = catmull_clark(working_mesh)
        n_cc += 1

        if validate_steps and not is_manifold(working_mesh):
            raise RuntimeError(f"Mesh became non-manifold after CC round {n_cc}")

    # ── CV tokens for vertex positions ────────────────────────────────
    # Map target positions to the working mesh's vertices by ordinal index.
    # If counts differ, emit min(len) CV tokens; remaining vertices keep
    # whatever position they have from the base/CC mesh.
    working_verts  = list(working_mesh.vertices.values())
    n_cv = min(len(target_positions), len(working_verts))

    for i in range(n_cv):
        x, y, z = target_positions[i]
        tokens.append(TopModToken(op='CV', pos=(
            quantize_coord(x, lo, hi, n_position_bins),
            quantize_coord(y, lo, hi, n_position_bins),
            quantize_coord(z, lo, hi, n_position_bins),
        )))

    tokens.append(TopModToken(op='EOS'))
    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Detokenize — token sequence → mesh
# ═══════════════════════════════════════════════════════════════════════════════

def _sta(mesh: DLFLMesh) -> DLFLMesh:
    stellate_all(mesh)
    return mesh


def _star(mesh: DLFLMesh) -> DLFLMesh:
    star_subdivide(mesh)
    return mesh


def _dome(mesh: DLFLMesh) -> DLFLMesh:
    dome_subdivide(mesh)
    return mesh


def _crust(mesh: DLFLMesh) -> DLFLMesh:
    # Mirror-pair face ordinals are deterministic (outer i ↔ inner F+i),
    # so hole punching is expressible with existing HDL(face1, face2)
    # tokens after CRUST.
    out, _pairs = create_crust(mesh)
    return out


# Zero-argument global remeshing opcodes → executor.
# LOOP / SQRT3 raise ValueError on non-triangular meshes (documented
# precondition); a generative model emitting them on invalid state gets a
# hard failure rather than a silent corruption.
_GLOBAL_OPS = {
    'DUAL':  dual,
    'DS':    doo_sabin,
    'STA':   _sta,
    'SIMP':  simplest_subdivide,
    'VC':    vertex_cutting,
    'LOOP':  loop_subdivide,
    'SQRT3': sqrt3_subdivide,
    'HONEY': honeycomb_subdivide,
    'STAR':  _star,
    'CCUT':  corner_cutting,
    'LSTYLE': loop_style_subdivide,
    'FRAC':  fractal_subdivide,
    'PENT':  pentagonal_subdivide,
    'PENT2': pentagonal2_subdivide,
    'D1264': dual1264_subdivide,
    'ROOT4': root4_subdivide,
    'CHKB':  checkerboard_remesh,
    'DSBC':  ds_bc_new_subdivide,
    'DOME':  _dome,
    'CRUST': _crust,
}

def detokenize(
    tokens:          List[TopModToken],
    n_position_bins: int   = 128,
    coord_lo:        float = DEFAULT_COORD_LO,
    coord_hi:        float = DEFAULT_COORD_HI,
    validate_steps:  bool  = False,
) -> DLFLMesh:
    """
    Execute a token sequence and return the resulting DLFLMesh.

    Always starts from make_icosahedron() (the implicit base).

    Invariant
    ---------
    After every structural token (HDL, IE, DE, CC), is_manifold() is True
    (guaranteed by the DLFL operators).

    Parameters
    ----------
    tokens           : Sequence produced by tokenize() (or hand-crafted).
    n_position_bins  : Must match the tokenize() call.
    coord_lo / hi    : Must match the tokenize() call.
    validate_steps   : If True, run is_manifold() after each structural op.

    Returns
    -------
    DLFLMesh — the reconstructed mesh.
    """
    mesh = make_icosahedron()

    # CV state — filled lazily on the first CV token
    cv_verts: Optional[List[Vertex]] = None
    cv_idx:   int = 0

    for token in tokens:
        op = token.op

        if op == 'EOS':
            break

        # ── Structural ops ────────────────────────────────────────────
        elif op == 'CC':
            mesh     = catmull_clark(mesh)
            cv_verts = None   # vertex list becomes invalid after CC
            cv_idx   = 0

            if validate_steps and not is_manifold(mesh):
                raise RuntimeError("Mesh became non-manifold after CC")

        elif op in _GLOBAL_OPS:
            mesh     = _GLOBAL_OPS[op](mesh)
            cv_verts = None   # vertex list becomes invalid
            cv_idx   = 0

            if validate_steps and not is_manifold(mesh):
                raise RuntimeError(f"Mesh became non-manifold after {op}")

        elif op == 'HDL':
            if token.corner1 is None or token.corner2 is None:
                raise ValueError(f"HDL token missing corner parameters: {token}")
            faces_list = list(mesh.faces.values())
            f1_ord, _  = token.corner1
            f2_ord, _  = token.corner2
            if f1_ord >= len(faces_list) or f2_ord >= len(faces_list):
                raise ValueError(
                    f"HDL ordinals ({f1_ord}, {f2_ord}) out of range "
                    f"(F={len(faces_list)})"
                )
            f1 = faces_list[f1_ord]
            f2 = faces_list[f2_ord]
            add_handle(mesh, f1, f2)

            if validate_steps and not is_manifold(mesh):
                raise RuntimeError(f"Mesh became non-manifold after HDL({f1_ord},{f2_ord})")

        elif op == 'IE':
            if token.corner1 is None or token.corner2 is None:
                raise ValueError(f"IE token missing corner parameters: {token}")
            faces_list    = list(mesh.faces.values())
            f1_ord, he1p  = token.corner1
            f2_ord, he2p  = token.corner2
            f1 = faces_list[f1_ord]
            f2 = faces_list[f2_ord]
            hes1 = f1.halfedges()
            hes2 = f2.halfedges()
            if he1p >= len(hes1) or he2p >= len(hes2):
                raise ValueError(
                    f"IE halfedge position out of range: "
                    f"he1p={he1p} (f1 deg {len(hes1)}), "
                    f"he2p={he2p} (f2 deg {len(hes2)})"
                )
            he1 = hes1[he1p]
            he2 = hes2[he2p]
            insert_edge(mesh, he1, he2)

            if validate_steps and not is_manifold(mesh):
                raise RuntimeError(
                    f"Mesh became non-manifold after IE(f{f1_ord}:h{he1p}, f{f2_ord}:h{he2p})"
                )

        elif op == 'DE':
            if token.edge_ord is None:
                raise ValueError(f"DE token missing edge_ord: {token}")
            edges_list = list(mesh.edges.values())
            if token.edge_ord >= len(edges_list):
                raise ValueError(
                    f"DE edge ordinal {token.edge_ord} out of range "
                    f"(E={len(edges_list)})"
                )
            e = edges_list[token.edge_ord]
            delete_edge(mesh, e)

            if validate_steps and not is_manifold(mesh):
                raise RuntimeError(f"Mesh became non-manifold after DE(e{token.edge_ord})")

        # ── Geometry op ───────────────────────────────────────────────
        elif op == 'CV':
            if token.pos is None:
                raise ValueError(f"CV token missing pos: {token}")
            if cv_verts is None:
                # Snapshot vertex list at first CV token (topology is frozen)
                cv_verts = list(mesh.vertices.values())
            if cv_idx < len(cv_verts):
                v = cv_verts[cv_idx]
                qx, qy, qz = token.pos
                v.x = dequantize_coord(qx, coord_lo, coord_hi, n_position_bins)
                v.y = dequantize_coord(qy, coord_lo, coord_hi, n_position_bins)
                v.z = dequantize_coord(qz, coord_lo, coord_hi, n_position_bins)
            cv_idx += 1   # always increment, even if we're past the end

        else:
            raise ValueError(f"Unknown token op: {op!r}")

    return mesh


# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary & integer encoding
# ═══════════════════════════════════════════════════════════════════════════════

def build_vocabulary(
    n_position_bins: int = 128,
    max_ordinal:     int = 65536,
) -> Dict[str, int]:
    """
    Build the token vocabulary mapping symbol names to integer IDs.

    Layout
    ------
    ID 0       : EOS
    ID 1       : CC
    ID 2       : CV    (opcode; followed by 3 × COORD tokens)
    ID 3       : IE    (opcode; followed by 4 × REF tokens)
    ID 4       : DE    (opcode; followed by 1 × REF token)
    ID 5       : HDL   (opcode; followed by 2 × REF tokens)
    ID 6..6+B-1: COORD_0 .. COORD_{B-1}  (quantized coordinates)
    ID 6+B ..  : REF_0 .. REF_{M-1}      (ordinal references)

    Total vocabulary size: 6 + n_position_bins + max_ordinal
    """
    vocab: Dict[str, int] = {}
    idx = 0

    for op in ('EOS', 'CC', 'CV', 'IE', 'DE', 'HDL'):
        vocab[op] = idx
        idx += 1

    for i in range(n_position_bins):
        vocab[f'COORD_{i}'] = idx
        idx += 1

    for i in range(max_ordinal):
        vocab[f'REF_{i}'] = idx
        idx += 1

    # Extension ops appended at the END so all pre-existing IDs
    # (EOS/CC/CV/IE/DE/HDL, COORD_*, REF_*) keep their values —
    # sequences and checkpoints encoded with the old vocabulary stay valid.
    for op in ('DUAL', 'DS', 'STA', 'SIMP', 'VC', 'LOOP', 'SQRT3',
               'HONEY', 'STAR', 'CCUT', 'LSTYLE', 'FRAC',
               'PENT', 'PENT2', 'D1264', 'ROOT4', 'CHKB', 'DSBC', 'DOME',
               'CRUST'):
        vocab[op] = idx
        idx += 1

    return vocab


def encode_sequence(
    tokens: List[TopModToken],
    vocab:  Dict[str, int],
) -> List[int]:
    """
    Encode a list of TopModTokens into a flat integer sequence.

    Encoding rules per token type
    -----------------------------
    EOS → [vocab['EOS']]
    CC  → [vocab['CC']]
    CV  → [vocab['CV'], vocab['COORD_qx'], vocab['COORD_qy'], vocab['COORD_qz']]
    HDL → [vocab['HDL'], vocab['REF_f1'], vocab['REF_f2']]
    IE  → [vocab['IE'],  vocab['REF_f1'], vocab['REF_h1'], vocab['REF_f2'], vocab['REF_h2']]
    DE  → [vocab['DE'],  vocab['REF_e']]
    """
    ids: List[int] = []

    for tok in tokens:
        op = tok.op

        if op == 'EOS':
            ids.append(vocab['EOS'])

        elif op == 'CC' or op in _GLOBAL_OPS:
            ids.append(vocab[op])

        elif op == 'CV':
            ids.append(vocab['CV'])
            qx, qy, qz = tok.pos
            ids.extend([
                vocab[f'COORD_{qx}'],
                vocab[f'COORD_{qy}'],
                vocab[f'COORD_{qz}'],
            ])

        elif op == 'HDL':
            ids.append(vocab['HDL'])
            f1_ord, _ = tok.corner1
            f2_ord, _ = tok.corner2
            ids.extend([vocab[f'REF_{f1_ord}'], vocab[f'REF_{f2_ord}']])

        elif op == 'IE':
            ids.append(vocab['IE'])
            f1_ord, he1p = tok.corner1
            f2_ord, he2p = tok.corner2
            ids.extend([
                vocab[f'REF_{f1_ord}'], vocab[f'REF_{he1p}'],
                vocab[f'REF_{f2_ord}'], vocab[f'REF_{he2p}'],
            ])

        elif op == 'DE':
            ids.append(vocab['DE'])
            ids.append(vocab[f'REF_{tok.edge_ord}'])

        else:
            raise ValueError(f"Unknown op: {op!r}")

    return ids


def decode_sequence(
    ids:       List[int],
    vocab_inv: Dict[int, str],
) -> List[TopModToken]:
    """
    Decode a flat integer sequence back to TopModTokens.

    *vocab_inv* should be {id: symbol_name}, the inverse of build_vocabulary().

    Raises ValueError on any unrecognised ID or malformed sequence.
    """
    OP_TOKENS  = {'EOS', 'CC', 'CV', 'IE', 'DE', 'HDL'} | set(_GLOBAL_OPS)
    tokens: List[TopModToken] = []
    it = iter(ids)

    def _next_symbol() -> str:
        try:
            return vocab_inv[next(it)]
        except StopIteration:
            raise ValueError("Unexpected end of integer sequence")

    def _next_coord() -> int:
        sym = _next_symbol()
        if not sym.startswith('COORD_'):
            raise ValueError(f"Expected COORD_*, got {sym!r}")
        return int(sym[6:])

    def _next_ref() -> int:
        sym = _next_symbol()
        if not sym.startswith('REF_'):
            raise ValueError(f"Expected REF_*, got {sym!r}")
        return int(sym[4:])

    for raw_id in it:
        sym = vocab_inv.get(raw_id)
        if sym is None:
            raise ValueError(f"Unknown vocab ID {raw_id}")

        if sym not in OP_TOKENS:
            # This ID is a param token that should have been consumed above
            raise ValueError(f"Unexpected param token {sym!r} at opcode position")

        if sym == 'EOS':
            tokens.append(TopModToken(op='EOS'))
            break

        elif sym == 'CC' or sym in _GLOBAL_OPS:
            tokens.append(TopModToken(op=sym))

        elif sym == 'CV':
            qx, qy, qz = _next_coord(), _next_coord(), _next_coord()
            tokens.append(TopModToken(op='CV', pos=(qx, qy, qz)))

        elif sym == 'HDL':
            f1_ord = _next_ref()
            f2_ord = _next_ref()
            tokens.append(TopModToken(op='HDL',
                                       corner1=(f1_ord, 0),
                                       corner2=(f2_ord, 0)))

        elif sym == 'IE':
            f1_ord, he1p = _next_ref(), _next_ref()
            f2_ord, he2p = _next_ref(), _next_ref()
            tokens.append(TopModToken(op='IE',
                                       corner1=(f1_ord, he1p),
                                       corner2=(f2_ord, he2p)))

        elif sym == 'DE':
            e_ord = _next_ref()
            tokens.append(TopModToken(op='DE', edge_ord=e_ord))

    return tokens


# ═══════════════════════════════════════════════════════════════════════════════
# Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def token_stats(tokens: List[TopModToken]) -> Dict[str, int]:
    """Return a summary of token type counts."""
    counts: Dict[str, int] = {}
    for tok in tokens:
        counts[tok.op] = counts.get(tok.op, 0) + 1
    return counts


def sequence_length(tokens: List[TopModToken]) -> int:
    """Return the total number of integer IDs produced by encode_sequence()."""
    length = 0
    for tok in tokens:
        if tok.op == 'CV':
            length += 4   # opcode + 3 coords
        elif tok.op in ('HDL',):
            length += 3   # opcode + 2 refs
        elif tok.op in ('IE',):
            length += 5   # opcode + 4 refs
        elif tok.op == 'DE':
            length += 2   # opcode + 1 ref
        else:
            length += 1   # EOS, CC and zero-arg global ops
    return length


# ═══════════════════════════════════════════════════════════════════════════════
# Vocabulary V2 — Propose-and-Optimize paradigm (Phase A')
# ═══════════════════════════════════════════════════════════════════════════════

# V2 operator list (29 total, IDs 1–29)
_V2_OPS: Tuple = (
    'CC', 'DS', 'HDL', 'IE', 'DE',
    'DUAL', 'STA', 'SIMP', 'VC', 'LOOP', 'SQRT3', 'HONEY', 'STAR',
    'CCUT', 'LSTYLE', 'FRAC', 'PENT', 'PENT2', 'D1264', 'ROOT4',
    'CHKB', 'DSBC', 'DOME', 'CRUST',
    'EXTRUDE_FACE', 'STELLATE', 'SUBDIVIDE_EDGE', 'SUBDIVIDE_FACE', 'CV',
)
_V2_OPS_SET: Set[str] = set(_V2_OPS)

# V2 base primitive tokens
_V2_BASES: Tuple = ('CUBE', 'TETRAHEDRON', 'ICOSAHEDRON')

# SEP token separates topology section from geometry (cage coords) section
_V2_SEP = 'SEP'


def build_vocabulary_v2(
    n_coord_bins: int = 256,
    n_ref:        int = 64,
) -> Dict[str, int]:
    """
    Build the 356-token V2 vocabulary for the propose-and-optimize paradigm.

    Layout
    ------
    0           : EOS
    1–29        : Operators (CC, DS, HDL, IE, DE, DUAL, STA, SIMP, VC, LOOP,
                  SQRT3, HONEY, STAR, CCUT, LSTYLE, FRAC, PENT, PENT2, D1264,
                  ROOT4, CHKB, DSBC, DOME, CRUST, EXTRUDE_FACE, STELLATE,
                  SUBDIVIDE_EDGE, SUBDIVIDE_FACE, CV)
    30–32       : BASE primitives (CUBE=30, TETRAHEDRON=31, ICOSAHEDRON=32)
    33          : SEP (topology ↔ geometry separator)
    34–289      : COORD_0 .. COORD_255  (256 quantized cage coordinate bins)
    290–353     : REF_0 .. REF_63       (face/edge ordinal references, up to 64)
    354         : BOS  (start token, decoder input only, never predicted)
    355         : PAD  (padding, ignored in loss)

    Total = 1 + 29 + 3 + 1 + 256 + 64 + 2 = 356
    """
    vocab: Dict[str, int] = {}
    idx = 0

    # EOS
    vocab['EOS'] = idx; idx += 1

    # 29 operators
    for op in _V2_OPS:
        vocab[op] = idx; idx += 1

    # 3 base primitives
    for base in _V2_BASES:
        vocab[f'BASE_{base}'] = idx; idx += 1

    # SEP
    vocab[_V2_SEP] = idx; idx += 1

    # COORD bins
    for i in range(n_coord_bins):
        vocab[f'COORD_{i}'] = idx; idx += 1

    # REF ordinals
    for i in range(n_ref):
        vocab[f'REF_{i}'] = idx; idx += 1

    # BOS and PAD
    vocab['BOS'] = idx; idx += 1
    vocab['PAD'] = idx; idx += 1

    return vocab


def encode_v2(
    base_name:    str,
    hdl_pairs:    List[Tuple[int, int]],
    op_names:     List[str],
    cage_verts:   "Any",  # [V_cage, 3] float array (numpy or sequence)
    vocab_v2:     Dict[str, int],
    coord_lo:     float = DEFAULT_COORD_LO,
    coord_hi:     float = DEFAULT_COORD_HI,
    n_coord_bins: int   = 256,
) -> List[int]:
    """
    Encode a V2 program + cage into a flat integer ID sequence.

    Parameters
    ----------
    base_name    : 'cube' | 'tetrahedron' | 'icosahedron'
    hdl_pairs    : list of (f1_ordinal, f2_ordinal) for each HDL op
    op_names     : list of linear/nonlinear operator names applied after HDL
    cage_verts   : [V_cage, 3] float array of (normalized) cage vertex positions
    vocab_v2     : vocabulary dict from build_vocabulary_v2()
    coord_lo/hi  : coordinate quantization range
    n_coord_bins : number of quantization bins (must match build_vocabulary_v2)

    Returns
    -------
    List[int] — flat token IDs, no BOS, includes EOS.

    Sequence layout:
        BASE_X  [HDL REF_f1 REF_f2]*  OP*  SEP  [Cx Cy Cz]*  EOS
    """
    ids: List[int] = []

    # BASE token
    base_key = f'BASE_{base_name.upper()}'
    if base_key not in vocab_v2:
        raise ValueError(f"Unknown base primitive: {base_name!r}")
    ids.append(vocab_v2[base_key])

    # HDL tokens (each HDL op uses 3 IDs: HDL + REF_f1 + REF_f2)
    for f1_ord, f2_ord in hdl_pairs:
        ids.append(vocab_v2['HDL'])
        ids.append(vocab_v2[f'REF_{f1_ord}'])
        ids.append(vocab_v2[f'REF_{f2_ord}'])

    # Linear / nonlinear op tokens (zero-argument in V2 topology format)
    for op in op_names:
        if op not in vocab_v2:
            raise ValueError(f"Unknown operator: {op!r}")
        ids.append(vocab_v2[op])

    # SEP: marks end of topology section
    ids.append(vocab_v2[_V2_SEP])

    # COORD tokens (flattened cage verts: x, y, z, x, y, z, …)
    V = cage_verts.shape[0]
    for i in range(V):
        x = float(cage_verts[i, 0])
        y = float(cage_verts[i, 1])
        z = float(cage_verts[i, 2])
        qx = quantize_coord(x, coord_lo, coord_hi, n_coord_bins)
        qy = quantize_coord(y, coord_lo, coord_hi, n_coord_bins)
        qz = quantize_coord(z, coord_lo, coord_hi, n_coord_bins)
        ids.append(vocab_v2[f'COORD_{qx}'])
        ids.append(vocab_v2[f'COORD_{qy}'])
        ids.append(vocab_v2[f'COORD_{qz}'])

    # EOS
    ids.append(vocab_v2['EOS'])

    return ids


def decode_v2(
    ids:          List[int],
    vocab_inv_v2: Dict[int, str],
) -> Dict:
    """
    Decode a flat V2 integer sequence into its components.

    Fault-tolerant: unknown IDs and malformed subsequences are silently skipped
    so that a partially correct model output still yields a parseable result.

    Parameters
    ----------
    ids          : flat integer sequence (may include BOS at position 0)
    vocab_inv_v2 : {id: symbol_name}, inverse of build_vocabulary_v2()

    Returns
    -------
    dict with keys:
      'base'       : str | None  (base primitive name, lowercase)
      'hdl_pairs'  : List[Tuple[int, int]]  (face ordinal pairs for HDL ops)
      'ops'        : List[str]              (operator names, in order)
      'coord_ints' : List[int]             (raw COORD bin values, flattened xyz)
    """
    base:        Optional[str]              = None
    hdl_pairs:   List[Tuple[int, int]]      = []
    ops:         List[str]                  = []
    coord_ints:  List[int]                  = []

    it = iter(ids)
    in_geometry = False

    for raw_id in it:
        sym = vocab_inv_v2.get(raw_id)
        if sym is None:
            continue

        if sym in ('BOS', 'PAD'):
            continue

        if sym == 'EOS':
            break

        if sym == _V2_SEP:
            in_geometry = True
            continue

        if in_geometry:
            if sym.startswith('COORD_'):
                coord_ints.append(int(sym[6:]))
        else:
            # Topology section
            if sym.startswith('BASE_'):
                base = sym[5:].lower()   # 'BASE_ICOSAHEDRON' → 'icosahedron'
            elif sym == 'HDL':
                # Consume next two tokens as REF ordinals
                s1 = vocab_inv_v2.get(next(it, None))
                s2 = vocab_inv_v2.get(next(it, None))
                f1 = int(s1[4:]) if s1 and s1.startswith('REF_') else 0
                f2 = int(s2[4:]) if s2 and s2.startswith('REF_') else 0
                hdl_pairs.append((f1, f2))
            elif sym in _V2_OPS_SET and sym not in ('HDL', 'EOS', 'CV'):
                ops.append(sym)

    return {
        'base':       base,
        'hdl_pairs':  hdl_pairs,
        'ops':        ops,
        'coord_ints': coord_ints,
    }
