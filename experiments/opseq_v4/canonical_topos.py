#!/usr/bin/env python3
"""
canonical_topos.py — Define the fixed topology catalog for v4 classification.

20 canonical topologies spanning 3 base primitives, genus 0-1, and common
subdivision operator sequences. Each entry specifies (base, genus, ops)
and can be built into a DLFL mesh via build_topo_mesh().

The classifier picks one of these 20 classes; vertex positions are then
refined via differentiable rendering at inference time.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from topmod.primitives     import make_cube, make_tetrahedron, make_icosahedron
from topmod.high_level_ops import add_handle
from topmod.subdivision    import catmull_clark
from topmod.remeshing      import (
    doo_sabin, dual1264_subdivide, root4_subdivide,
    checkerboard_remesh, ds_bc_new_subdivide, corner_cutting,
)
from topmod.diffgeo        import mesh_to_arrays, _fan_triangulate


# ── Canonical topology catalog ───────────────────────────────────────────────

CANONICAL_TOPOS = [
    {'id': 0,  'base': 'tet',  'genus': 0, 'ops': [],               'label': 'tet_g0_4v'},
    {'id': 1,  'base': 'cube', 'genus': 0, 'ops': [],               'label': 'cube_g0_8v'},
    {'id': 2,  'base': 'ico',  'genus': 0, 'ops': [],               'label': 'ico_g0_12v'},
    {'id': 3,  'base': 'tet',  'genus': 0, 'ops': ['CC'],           'label': 'tet_g0_CC_14v'},
    {'id': 4,  'base': 'cube', 'genus': 0, 'ops': ['DS'],           'label': 'cube_g0_DS_24v'},
    {'id': 5,  'base': 'cube', 'genus': 0, 'ops': ['CC'],           'label': 'cube_g0_CC_26v'},
    {'id': 6,  'base': 'cube', 'genus': 0, 'ops': ['ROOT4'],        'label': 'cube_g0_ROOT4_32v'},
    {'id': 7,  'base': 'tet',  'genus': 0, 'ops': ['CC', 'CC'],     'label': 'tet_g0_CC2_50v'},
    {'id': 8,  'base': 'cube', 'genus': 0, 'ops': ['DSBC'],         'label': 'cube_g0_DSBC_56v'},
    {'id': 9,  'base': 'cube', 'genus': 0, 'ops': ['CHKB'],         'label': 'cube_g0_CHKB_56v'},
    {'id': 10, 'base': 'ico',  'genus': 0, 'ops': ['DS'],           'label': 'ico_g0_DS_60v'},
    {'id': 11, 'base': 'ico',  'genus': 0, 'ops': ['CC'],           'label': 'ico_g0_CC_62v'},
    {'id': 12, 'base': 'cube', 'genus': 0, 'ops': ['CC', 'DS'],     'label': 'cube_g0_CCDS_96v'},
    {'id': 13, 'base': 'cube', 'genus': 0, 'ops': ['CC', 'CC'],     'label': 'cube_g0_CC2_98v'},
    {'id': 14, 'base': 'cube', 'genus': 0, 'ops': ['ROOT4', 'D1264'], 'label': 'cube_g0_R4D_192v'},
    {'id': 15, 'base': 'ico',  'genus': 0, 'ops': ['CC', 'DS'],     'label': 'ico_g0_CCDS_240v'},
    {'id': 16, 'base': 'ico',  'genus': 0, 'ops': ['CC', 'CC'],     'label': 'ico_g0_CC2_242v'},
    {'id': 17, 'base': 'ico',  'genus': 1, 'ops': [],               'label': 'ico_g1_12v'},
    {'id': 18, 'base': 'ico',  'genus': 1, 'ops': ['CC'],           'label': 'ico_g1_CC_66v'},
    {'id': 19, 'base': 'ico',  'genus': 1, 'ops': ['CC', 'CC'],     'label': 'ico_g1_CC2_264v'},
]

N_CLASSES = len(CANONICAL_TOPOS)  # 20


# ── Primitive factory ────────────────────────────────────────────────────────

_PRIM_FNS = {
    'tet':  make_tetrahedron,
    'cube': make_cube,
    'ico':  make_icosahedron,
}

# ── Operator dispatch ────────────────────────────────────────────────────────

_OP_FNS = {
    'CC':    catmull_clark,
    'DS':    doo_sabin,
    'D1264': dual1264_subdivide,
    'ROOT4': root4_subdivide,
    'CHKB':  checkerboard_remesh,
    'DSBC':  ds_bc_new_subdivide,
    'CCUT':  corner_cutting,
}


# ── Build mesh from a canonical entry ────────────────────────────────────────

def build_topo_mesh(
    topo_entry: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a DLFL mesh from a canonical topology entry.

    Parameters
    ----------
    topo_entry : dict with keys 'base', 'genus', 'ops'

    Returns
    -------
    positions_np : [V, 3] float64 vertex positions
    tris_np      : [T, 3] int32   triangle indices (fan-triangulated)
    """
    base  = topo_entry['base']
    genus = topo_entry['genus']
    ops   = topo_entry['ops']

    mesh = _PRIM_FNS[base]()

    # Apply HDL ops for genus > 0
    # For icosahedron genus=1: use face ordinals (0, 5)
    if genus >= 1:
        faces_list = list(mesh.faces.values())
        if len(faces_list) > 5:
            add_handle(mesh, faces_list[0], faces_list[5])

    # Apply subdivision ops
    for op_name in ops:
        fn = _OP_FNS.get(op_name)
        if fn is None:
            raise ValueError(f"Unknown op: {op_name}")
        mesh = fn(mesh)

    # Extract arrays
    positions, faces = mesh_to_arrays(mesh)
    positions_np = np.array(positions, dtype=np.float64)
    tris = _fan_triangulate(faces)
    tris_np = np.array(tris, dtype=np.int32)

    return positions_np, tris_np


# ── Self-test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print(f"Canonical topologies: {N_CLASSES}")
    for entry in CANONICAL_TOPOS:
        try:
            verts, tris = build_topo_mesh(entry)
            print(f"  [{entry['id']:2d}] {entry['label']:25s}  "
                  f"V={verts.shape[0]:5d}  T={tris.shape[0]:5d}")
        except Exception as exc:
            print(f"  [{entry['id']:2d}] {entry['label']:25s}  FAILED: {exc}")
