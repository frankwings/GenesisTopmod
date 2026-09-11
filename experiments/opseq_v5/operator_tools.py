"""
operator_tools.py — Phase 1: wrap topmod_extrude_cluster as the VLM operator.

The VLM loop calls execute_operator(op_name, mesh_state, **kwargs) → new mesh.

Phase 1 exposes a single operator:
  "extrude"  → topmod_extrude_cluster(cluster_faces, extrude_dir, dist)

Future phases may add: "split_edge", "smooth_region", "delete_face", etc.

Usage
-----
    from operator_tools import execute_operator, OPERATOR_DESCRIPTIONS

    new_verts, new_tris, old_V = execute_operator(
        "extrude",
        verts_np     = verts_np,
        tris_np      = tris_np,
        cluster_faces = cluster,
        extrude_dir  = extrude_dir,
        dist         = 0.05,
    )
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# Import the DLFL operator from eval_extrude_v3
try:
    from eval_extrude_v3 import topmod_extrude_cluster, dlfl_boundary_stats
except ImportError:
    from experiments.opseq_v5.eval_extrude_v3 import (
        topmod_extrude_cluster, dlfl_boundary_stats
    )


# ── Registry ─────────────────────────────────────────────────────────────────

#: Human-readable descriptions of each operator (for VLM prompt construction).
OPERATOR_DESCRIPTIONS: Dict[str, str] = {
    "extrude": (
        "Extrude a cluster of faces outward to add missing geometry volume. "
        "Use when a region of the silhouette is consistently missing across "
        "multiple views — the rendered shape is 'too flat' in that area."
    ),
}


def list_operators() -> list:
    """Return sorted list of available operator names."""
    return sorted(OPERATOR_DESCRIPTIONS.keys())


def execute_operator(
    op_name: str,
    verts_np: np.ndarray,
    tris_np:  np.ndarray,
    **kwargs: Any,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Dispatch to the named operator and return (new_verts, new_tris, old_V).

    Parameters
    ----------
    op_name : str
        One of the keys in OPERATOR_DESCRIPTIONS.
    verts_np : [V, 3] float32/64
        Current vertex positions.
    tris_np : [F, 3] int32
        Current triangle indices.
    **kwargs :
        Operator-specific parameters (see each operator's docstring below).

    Returns
    -------
    new_verts : [V', 3] float64
    new_tris  : [F', 3] int32
    old_V     : int
        Vertex count before the operation (new vertices start at old_V).

    Raises
    ------
    ValueError
        If op_name is not recognised.
    AssertionError
        If the resulting mesh has boundary or non-manifold edges.
    """
    if op_name == "extrude":
        return _op_extrude(verts_np, tris_np, **kwargs)
    else:
        raise ValueError(
            f"Unknown operator {op_name!r}. "
            f"Available: {list_operators()}"
        )


# ── Operator implementations ─────────────────────────────────────────────────

def _op_extrude(
    verts_np:      np.ndarray,
    tris_np:       np.ndarray,
    cluster_faces: np.ndarray,
    extrude_dir:   np.ndarray,
    dist:          float = 0.05,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Extrude a cluster of faces via TopMod DLFL.

    Parameters
    ----------
    cluster_faces : [K] int32
        Face indices to extrude.
    extrude_dir : [3] float64
        Unit direction for the extrusion.
    dist : float
        Extrusion distance in mesh units.

    Returns
    -------
    (new_verts, new_tris, old_V)
    """
    cluster_faces = np.asarray(cluster_faces, dtype=np.int32)
    extrude_dir   = np.asarray(extrude_dir,   dtype=np.float64)
    dn = np.linalg.norm(extrude_dir)
    if dn > 1e-8:
        extrude_dir = extrude_dir / dn
    else:
        extrude_dir = np.array([0., 1., 0.])

    dist = float(abs(dist))
    if dist < 1e-6:
        dist = 0.02   # safety floor

    new_verts, new_tris, old_V = topmod_extrude_cluster(
        verts_np      = verts_np.astype(np.float64),
        tris_np       = tris_np.astype(np.int32),
        cluster_faces = cluster_faces,
        extrude_dir   = extrude_dir,
        dist          = dist,
    )

    # Sanity: verify zero boundary edges on result
    n_bnd, n_nm = dlfl_boundary_stats(new_verts, new_tris)
    if n_nm > 0:
        raise AssertionError(
            f"extrude produced {n_nm} non-manifold edges — mesh corrupted"
        )
    if n_bnd > 0:
        raise AssertionError(
            f"extrude produced {n_bnd} boundary edges — mesh is open"
        )

    return new_verts, new_tris, old_V
