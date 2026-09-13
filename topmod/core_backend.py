"""
C++ DLFL backend wrapper.  Exposes the same array-level signatures as the
Python callers (flip_sweep, collapse_short_edges, subdivide_faces, etc.).

Import guard: if topmod_core not built, raise ImportError with instructions.

Build the extension first:
    cd /path/to/GenesisTopmod/topmod/cpp
    python3 setup.py build_ext --inplace
"""
import os, sys
import numpy as np

# Locate .so in this directory or in cpp/ subdir
_HERE = os.path.dirname(os.path.abspath(__file__))
_CPP_DIR = os.path.join(_HERE, 'cpp')
for _d in (_HERE, _CPP_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

try:
    import topmod_core as _tc
except ImportError as _e:
    raise ImportError(
        "topmod_core C++ extension not found. Build it first:\n"
        "  cd topmod/cpp && python3 setup.py build_ext --inplace\n"
        f"Original error: {_e}"
    )


def flip_sweep(V, F, passes=4, fold_cos=0.0):
    """V unchanged, returns (V, F2, n_flips)."""
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    V2, F2, n = _tc.flip_sweep(V, F, passes, fold_cos)
    return V, np.asarray(F2, dtype=np.int64), n


def collapse_short_edges(V, F, ratio=0.3, max_n=400, thr_abs=None, vthr=None):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    thr = thr_abs if thr_abs is not None else -1.0
    vthr_arr = np.asarray(vthr, dtype=np.float64) if vthr is not None else np.empty(0)
    V2, F2, n = _tc.collapse_short_edges(V, F, ratio, max_n, thr, vthr_arr)
    return np.asarray(V2, dtype=np.float64), np.asarray(F2, dtype=np.int64), n


def subdivide_faces(V, F, fids, expand_ring=True):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    fids = list(int(x) for x in fids)
    V2, F2, n_split = _tc.subdivide_faces(V, F, fids, expand_ring)
    return np.asarray(V2, dtype=np.float64), np.asarray(F2, dtype=np.int64), n_split


def catmull_clark(V, polys):
    V = np.asarray(V, dtype=np.float64)
    polys = [list(int(x) for x in p) for p in polys]
    V2, polys2 = _tc.catmull_clark(V, polys)
    return np.asarray(V2, dtype=np.float64), polys2


def triangulate_all(V, polys):
    V = np.asarray(V, dtype=np.float64)
    polys = [list(int(x) for x in p) for p in polys]
    tris = _tc.triangulate_all(V, polys)
    return np.asarray(tris, dtype=np.int64)


def add_handle(V, F, fi, fj):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    V2, F2 = _tc.add_handle(V, F, int(fi), int(fj))
    return np.asarray(V2, dtype=np.float64), np.asarray(F2, dtype=np.int64)


def check_watertight(F):
    F = np.asarray(F, dtype=np.int64)
    ok, n_bad = _tc.check_watertight(F)
    return bool(ok), int(n_bad)


def euler_genus(V, F):
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    return int(_tc.euler_genus(V, F))
