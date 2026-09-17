#!/usr/bin/env python3
"""membrane_locate.py — Boss's formulation (2026-09-16): find tunnel membranes ON THE DR MESH.
The space-carved hull never intersects the object, so every DR vertex that lies clearly OUTSIDE the hull
(more than MARGIN voxels) is spanning air = it belongs to a membrane / a mouth of an unopened chamber.
  1. membrane vertices: HF.dist(v) > MARGIN * pitch   (fitting slop near the hull is inside the margin)
  2. patches: connected components of membrane vertices over mesh edges; a patch whose vertex normals point both
     ways (thin membrane = top skin + bottom skin pressed together) is split by normal sign into two patches
  3. one representative face per patch: the all-membrane face deepest in air
  4. pairing: two patches belong to the same tunnel iff the segment between their faces runs through hull AIR
     and through mesh INTERIOR (occupancy queries) -> add_handle(face_i, face_j) digs the tunnel
Returns [(fi, fj, ci, cj, key)] best pair first; the caller adds ONE handle, verifies by DR, re-detects."""
import os, numpy as np, torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

MARGIN = float(os.environ.get("MEMB_MARGIN", "2.0"))     # voxels outside the hull to count as membrane
AIR_FRAC = float(os.environ.get("MEMB_AIR", "0.8"))       # fraction of the segment that must be hull air
IN_FRAC = float(os.environ.get("MEMB_IN", "0.8"))         # fraction of the segment that must be inside the mesh
MIN_VERTS = int(os.environ.get("MEMB_MIN_VERTS", "3"))

def _vertex_normals(V, F):
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); vn = np.zeros_like(V)
    for k in range(3): np.add.at(vn, F[:, k], fn)
    return vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12)

_BARY = None
def _bary(k=4):
    """barycentric sample grid inside a triangle (k+1)(k+2)/2 points, edges excluded"""
    pts = [(i / k, j / k, 1 - (i + j) / k) for i in range(1, k) for j in range(1, k - i)]
    return np.array(pts if pts else [(1/3, 1/3, 1/3)], float)


def _voxel_blocks(V, F, HF, res=128, pad=6, d=2, device="cuda"):
    """Mesh material where the hull says air, as voxel blocks with k = genus gain when the block is removed.
    mesh_solid = fill(surface voxels | occupancy); M = mesh_solid & ~dilate(hull, d)."""
    import open3d as o3d
    from scipy import ndimage
    from hull_locate import clean_hull, downsample, genus_solid, S26
    hs = np.pad(downsample(clean_hull(np.asarray(HF.hull).astype(bool)), res), pad)
    lo, hi = np.asarray(HF.lo, float), np.asarray(HF.hi, float); N = hs.shape[0]; pitch = (hi - lo) / (res - 1)
    def v2w(v): return lo + (np.asarray(v, float) - pad) / (res - 1) * (hi - lo)
    def w2v(w): return (np.asarray(w, float) - lo) / (hi - lo) * (res - 1) + pad
    surf = np.zeros_like(hs)
    tri = V[F]; e = np.max(np.linalg.norm(tri[:, [1, 2, 2]] - tri[:, [0, 0, 1]], axis=2), axis=1) / pitch.min()
    for t, ne in zip(tri, np.ceil(e * 2).astype(int) + 1):
        a = np.linspace(0, 1, ne); A, B = np.meshgrid(a, a); m = A + B <= 1; A, B = A[m], B[m]
        P = t[0] + A[:, None] * (t[1] - t[0]) + B[:, None] * (t[2] - t[0])
        q = np.clip(np.round(w2v(P)).astype(int), 0, N - 1); surf[q[:, 0], q[:, 1], q[:, 2]] = True
    rs = o3d.t.geometry.RaycastingScene(); rs.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32))))
    g = np.indices((N, N, N)).reshape(3, -1).T
    occ = rs.compute_occupancy(o3d.core.Tensor(v2w(g).astype(np.float32))).numpy().reshape(N, N, N) > 0.5
    mesh_solid = ndimage.binary_fill_holes(surf | occ); g0 = genus_solid(mesh_solid)
    M = mesh_solid & ~ndimage.binary_dilation(hs, S26, iterations=d)
    lab, n = ndimage.label(M, structure=S26); sizes = np.bincount(lab.ravel())[1:] if n else np.zeros(0, int)
    blocks = []
    for c in np.argsort(-sizes):
        if sizes[c] < 30: break
        C = lab == (c + 1); k = genus_solid(mesh_solid & ~ndimage.binary_dilation(C, S26, iterations=1)) - g0
        idx = np.argwhere(C); blocks.append((idx, int(k), v2w(idx.mean(0))))
    return blocks, v2w, float(pitch.max())

def find_tunnel_by_membranes(V, F, HF, prev_handles=(), g_target=None, r_dedup=0.3, log=print, device="cuda"):
    """Face-level: the coarse DR mesh spans a tunnel with a few LARGE faces whose vertices sit on the rim (inside the
    margin) -> test face INTERIOR samples, not vertices. A face is membrane if the median of its interior samples is
    > MARGIN voxels outside the hull."""
    import open3d as o3d
    V = np.asarray(V, float); F = np.asarray(F, np.int64); n = len(V); nf = len(F)
    B = _bary(int(os.environ.get("MEMB_BARY", "6")))                              # 10 interior samples per face
    P = np.einsum("sk,fkc->fsc", B, V[F]).reshape(-1, 3)
    with torch.no_grad(): d = HF.dist(torch.tensor(P, dtype=torch.float32, device=device)).cpu().numpy().reshape(nf, len(B))
    pitch = float(HF.pitch); fdist = np.median(d, axis=1); memb_f = fdist > MARGIN * pitch
    log(f"[memb] {int(memb_f.sum())}/{nf} faces span air (> {MARGIN:g} vox outside the hull)")
    if memb_f.sum() < 1: return []
    # face adjacency over shared edges, restricted to membrane faces
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1); fid = np.tile(np.arange(nf), 3)
    key = E[:, 0] * n + E[:, 1]; order = np.argsort(key); key, fid = key[order], fid[order]
    same = key[1:] == key[:-1]; fa, fb = fid[:-1][same], fid[1:][same]
    keep = memb_f[fa] & memb_f[fb]
    A = coo_matrix((np.ones(keep.sum()), (fa[keep], fb[keep])), shape=(nf, nf)); _, lab = connected_components(A, directed=False)
    lab[~memb_f] = -1
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    tri_cen = V[F].mean(1)
    patches = []                      # (face ids, representative face, centroid, normal)
    for c in np.unique(lab[lab >= 0]):
        ids = np.where(lab == c)[0]
        nm = fn[ids].mean(0); nm /= np.linalg.norm(nm) + 1e-12; side = fn[ids] @ nm
        groups = [ids[side >= 0], ids[side < 0]] if min((side < 0).sum(), (side >= 0).sum()) >= max(1, 0.3 * len(ids)) else [ids]   # split only when both skins are substantial
        for g in groups:
            f0 = int(g[np.argmax(fdist[g])])
            patches.append((g, f0, tri_cen[f0], fn[g].mean(0) / (np.linalg.norm(fn[g].mean(0)) + 1e-12)))
    log(f"[memb] {len(patches)} patches (faces): {[len(p[0]) for p in patches]}")
    if len(patches) < 2: return []
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32))))
    # voxel blocks: keep only patches sitting on a block whose removal opens a cycle (k>=1); fitting-slop patches touch none
    from scipy.spatial import cKDTree
    blocks, v2w, pmax = _voxel_blocks(V, F, HF, device=device)
    log(f"[memb] voxel blocks (size, k): {[(len(b[0]), b[1]) for b in blocks]}")
    patch_block = []
    for (ids, f0, c, nm) in patches:
        best = (-1, 1e9)
        for bi, (idx, k, cen_b) in enumerate(blocks):
            if k < 1: continue
            dmin = cKDTree(v2w(idx)).query(c)[0]
            if dmin <= 3.0 * pmax and dmin < best[1]: best = (bi, dmin)
        patch_block.append(best[0])
    log(f"[memb] patch -> block: {patch_block}")
    any_block = any(pb >= 0 for pb in patch_block)
    ts = np.linspace(0.06, 0.94, 23)
    cands = []
    for a in range(len(patches)):
        for b in range(a + 1, len(patches)):
            ca, cb = patches[a][2], patches[b][2]; sep = float(np.linalg.norm(cb - ca))
            if any_block and not (patch_block[a] >= 0 and patch_block[a] == patch_block[b]): continue   # same k>=1 block only
            if sep < 1.5 * pitch and float(patches[a][3] @ patches[b][3]) < -0.3:      # the two skins of one pinched membrane
                cands.append((min(len(patches[a][0]), len(patches[b][0])), a, b, 1.0, 1.0, "pinched")); continue
            seg = ca[None, :] + ts[:, None] * (cb - ca)[None, :]
            with torch.no_grad(): air = (HF.dist(torch.tensor(seg, dtype=torch.float32, device=device)).cpu().numpy() > 0.5 * pitch).mean()
            inside = (scene.compute_occupancy(o3d.core.Tensor(seg.astype(np.float32))).numpy() > 0.5).mean()
            if air >= AIR_FRAC and inside >= IN_FRAC:
                cands.append((min(len(patches[a][0]), len(patches[b][0])), a, b, air, inside, f"through block{patch_block[a]}"))
    cands.sort(key=lambda t: (-t[0], np.linalg.norm(patches[t[1]][2] - patches[t[2]][2])))
    out = []
    for score, a, b, air, inside, how in cands:
        fi, fj = patches[a][1], patches[b][1]; ci, cj = tri_cen[fi], tri_cen[fj]; mid = 0.5 * (ci + cj)
        key = ["memb", [round(float(x), 3) for x in mid]]
        dup = any(isinstance(h, dict) and (h.get("blob") == key or (h.get("mid") is not None and np.linalg.norm(np.asarray(h["mid"], float) - mid) < r_dedup)) for h in prev_handles)
        if dup: log(f"[memb] pair patches {a},{b} at {np.round(mid,3)}: dedup skip"); continue
        log(f"[memb] pair patches {a}({len(patches[a][0])}f),{b}({len(patches[b][0])}f): faces {fi},{fj} sep={np.linalg.norm(cj-ci):.3f} air={air:.2f} inside={inside:.2f} {how}")
        out.append((fi, fj, ci, cj, key))
    log(f"[memb] {len(cands)} pairs through hull air, {len(out)} after dedup")
    find_tunnel_by_membranes.last = dict(memb_f=memb_f, lab=lab, patches=patches, cands=cands, fdist=fdist)
    return out
