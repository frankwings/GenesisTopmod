"""DLFL edge-flip untangling as a reusable module (factored from phase3d).

flip = delete_edge (merge the two triangles into a quad) + insert_edge along
the other diagonal. V/E/F unchanged, 2-manifold preserved (DLFL closure).
Guards: both faces triangles, apexes distinct, new diagonal not already an
edge (no duplicate edges), post-flip triangles non-degenerate and less folded.
"""
import os, tempfile
import numpy as np
from topmod.io import from_obj, to_triangle_arrays
from topmod.operators import insert_edge, delete_edge


def _tri_normal(a, b, c):
    n = np.cross([b.x-a.x, b.y-a.y, b.z-a.z], [c.x-a.x, c.y-a.y, c.z-a.z])
    l = np.linalg.norm(n)
    return (n / l if l > 1e-14 else None), l


def try_flip(mesh, edge, fold_cos=0.0):
    he0, he1 = edge.he0, edge.he1
    fa, fb = he0.face, he1.face
    if fa is None or fb is None or fa.degree() != 3 or fb.degree() != 3:
        return False
    v0, v1 = he0.origin, he1.origin
    c = he0.prev.origin
    d = he1.prev.origin
    if c is d or c is v0 or c is v1 or d is v0 or d is v1:
        return False
    na, la = _tri_normal(v0, v1, c); nb, lb = _tri_normal(v1, v0, d)
    if na is None or nb is None: return False
    cur = float(na @ nb)
    if cur >= fold_cos: return False
    for h in c.outgoing_halfedges():
        if h.twin is not None and h.twin.origin is d: return False
    n1, l1 = _tri_normal(c, v0, d); n2, l2 = _tri_normal(d, v1, c)
    if n1 is None or n2 is None or l1 < 1e-3 * (la + lb) or l2 < 1e-3 * (la + lb):
        return False
    if float(n1 @ n2) <= cur + 1e-6: return False
    merged = delete_edge(mesh, edge)
    hc = hd = None
    for h in merged.halfedges():
        if h.origin is c: hc = h
        if h.origin is d: hd = h
    if hc is None or hd is None:
        raise RuntimeError("flip: corners lost after delete_edge")
    insert_edge(mesh, hc, hd)
    return True


def flip_sweep(V, Fa, passes=4, fold_cos=0.0):
    """Returns V (unchanged values, DLFL order), new faces, total flips."""
    if os.environ.get("GENERIC_OPS") == "1":
        from generic_ops import flip_sweep_np
        return flip_sweep_np(V, Fa, passes, fold_cos)
    V = np.asarray(V, float); Fa = np.asarray(Fa, np.int64)
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, z in V: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    try:
        mesh = from_obj(path)
    finally:
        os.unlink(path)
    total = 0
    for _ in range(passes):
        n = 0
        for e in list(mesh.edges.values()):
            if e.id not in mesh.edges: continue
            if try_flip(mesh, e, fold_cos): n += 1
        total += n
        if n == 0: break
    if total == 0:
        return V, Fa, 0
    vv, ff = to_triangle_arrays(mesh)
    V2 = np.asarray(vv, float)
    assert np.allclose(V2, V, atol=1e-9), "flip must not move vertices"
    return V, np.asarray(ff, np.int64), total


def collapse_short_edges(V, Fa, ratio=0.3, max_n=400):
    """DLFL collapse_edge_tri on edges shorter than ratio * mean edge, shortest
    first (link-condition guarded, Euler preserved). Residual self-intersections
    after flips were 46% tiny faces crowded together (6v tail region) -- an
    overcrowded triangulation, fixed by remeshing-style short-edge collapse.
    Returns V2, F2, n_collapsed, keep_index (old vertex idx surviving, or -1)."""
    if os.environ.get("GENERIC_OPS") == "1":
        from generic_ops import collapse_short_edges_np
        return collapse_short_edges_np(V, Fa, ratio, max_n)
    from topmod.high_level_ops import collapse_edge_tri
    V = np.asarray(V, float); Fa = np.asarray(Fa, np.int64)
    E = np.concatenate([Fa[:, [0, 1]], Fa[:, [1, 2]], Fa[:, [2, 0]]])
    el = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    thr = ratio * el.mean()
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, z in V: fh.write(f"v {x} {y} {z}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    try:
        mesh = from_obj(path)
    finally:
        os.unlink(path)
    def elen(e):
        a, b = e.he0.origin, e.he1.origin
        return (a.x-b.x)**2 + (a.y-b.y)**2 + (a.z-b.z)**2
    n = 0
    for e in sorted(list(mesh.edges.values()), key=elen):
        if e.id not in mesh.edges: continue
        if elen(e) > thr * thr: break
        if collapse_edge_tri(mesh, e) is not None:
            n += 1
            if n >= max_n: break
    if n == 0:
        return V, Fa, 0
    vv, ff = to_triangle_arrays(mesh)
    return np.asarray(vv, float), np.asarray(ff, np.int64), n


def tangential_smooth(V, Fa, iters=1, lam=0.2):
    V = np.asarray(V, float).copy(); nv = len(V)
    src = np.concatenate([Fa[:, 0], Fa[:, 1], Fa[:, 2], Fa[:, 1], Fa[:, 2], Fa[:, 0]])
    dst = np.concatenate([Fa[:, 1], Fa[:, 2], Fa[:, 0], Fa[:, 0], Fa[:, 1], Fa[:, 2]])
    deg = np.bincount(src, minlength=nv).astype(float)
    for _ in range(iters):
        cen = np.zeros_like(V); np.add.at(cen, src, V[dst]); cen /= np.maximum(deg, 1)[:, None]
        fn = np.cross(V[Fa[:, 1]] - V[Fa[:, 0]], V[Fa[:, 2]] - V[Fa[:, 0]])
        vn = np.zeros_like(V)
        for k in range(3): np.add.at(vn, Fa[:, k], fn)
        vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
        d = cen - V
        d -= (d * vn).sum(1, keepdims=True) * vn
        V += lam * d
    return V


def si_faces(V, Fa):
    import open3d as o3d
    om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, float)),
                                   o3d.utility.Vector3iVector(np.asarray(Fa, np.int32)))
    pairs = np.asarray(om.get_self_intersecting_triangles())
    return int(len(np.unique(pairs))) if len(pairs) else 0


def fold_frac(V, Fa, cos_thr=-0.5):
    import collections
    V = np.asarray(V, float); Fa = np.asarray(Fa, np.int64)
    n = np.cross(V[Fa[:, 1]] - V[Fa[:, 0]], V[Fa[:, 2]] - V[Fa[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ef = collections.defaultdict(list)
    for i, (a, b, c) in enumerate(Fa):
        for e in ((a, b), (b, c), (c, a)):
            ef[(min(e), max(e))].append(i)
    cs = np.array([n[f[0]] @ n[f[1]] for f in ef.values() if len(f) == 2])
    return float((cs < cos_thr).mean())
