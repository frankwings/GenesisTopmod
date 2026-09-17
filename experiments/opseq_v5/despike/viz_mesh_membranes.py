#!/usr/bin/env python3
"""viz_mesh_membranes.py <shape> <cc3p4.npz> <out.png>
THROWAWAY VIZ for the Boss (2026-09-16): find tunnel-blocking membranes ON THE MESH, hull used only as air classifier.
  mesh_solid = fill(surface voxels | occupancy)  (surface rasterised so thin membranes are >= 1 voxel)
  M = mesh_solid & ~dilate(hull, d)              (mesh material where the hull says air, minus fitting slop)
  per component C of M:  k = genus(mesh_solid \\ dilate(C,1)) - genus(mesh_solid)   (k>=1: removing C opens k cycles)
  mouths(C) = components of (shell(C) & mesh_air)  = where the block meets the outside air"""
import sys, os, numpy as np, torch
sys.path[:0] = [".", "despike", "/home/kingy/Projects/Genesis/GenesisTopmod"]; os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
import open3d as o3d, nvdiffrast.torch as dr
from scipy import ndimage
from skimage import measure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import run_64v
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
from hull_field import build_vote_hull
from hull_locate import clean_hull, downsample, genus_solid, S26
shape, npz, out = sys.argv[1], sys.argv[2], sys.argv[3]; RES = 128; PAD = 6; D = int(os.environ.get("MEMB_D", "2"))
ctx = dr.RasterizeCudaContext()
gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{shape}.obj")); gvn = normalize_to_range(gv)
mvps, views = run_64v.star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
HF = build_vote_hull(ctx, mvps, gvn, gf, None, "cuda", nres=256, hires=512, vote=2)
hs = np.pad(downsample(clean_hull(np.asarray(HF.hull).astype(bool)), RES), PAD)
lo, hi = np.asarray(HF.lo, float), np.asarray(HF.hi, float)
def v2w(v): return lo + (np.asarray(v, float) - PAD) / (RES - 1) * (hi - lo)
def w2v(w): return (np.asarray(w, float) - lo) / (hi - lo) * (RES - 1) + PAD
z = np.load(npz); V, F = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
N = hs.shape[0]; pitch = (hi - lo) / (RES - 1)
# surface rasterisation: sample each triangle at ~0.5 voxel spacing
surf = np.zeros_like(hs)
for tri in V[F]:
    e = max(np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[0]), np.linalg.norm(tri[2] - tri[1])) / pitch.min()
    n = int(np.ceil(e * 2)) + 1; a = np.linspace(0, 1, n)
    A, B = np.meshgrid(a, a); m = A + B <= 1; A, B = A[m], B[m]
    P = tri[0] + A[:, None] * (tri[1] - tri[0]) + B[:, None] * (tri[2] - tri[0])
    q = np.clip(np.round(w2v(P)).astype(int), 0, N - 1); surf[q[:, 0], q[:, 1], q[:, 2]] = True
scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32))))
g = np.indices((N, N, N)).reshape(3, -1).T
occ = scene.compute_occupancy(o3d.core.Tensor(v2w(g).astype(np.float32))).numpy().reshape(N, N, N) > 0.5
mesh_solid = ndimage.binary_fill_holes(surf | occ); mesh_air = ~mesh_solid
g_mesh_vox = genus_solid(mesh_solid); E = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0)); g_mesh = (2 - (len(V) - E + len(F))) // 2
hull_near = ndimage.binary_dilation(hs, S26, iterations=D)
M = mesh_solid & ~hull_near
lab, n = ndimage.label(M, structure=S26); sizes = np.bincount(lab.ravel())[1:]
print(f"[{shape}] hull genus {genus_solid(hs)} | mesh genus {g_mesh} (voxel {g_mesh_vox}) | M = {M.sum()} vox in {n} comps (>=30: {(sizes>=30).sum()})")
comps = []
for c in np.argsort(-sizes):
    if sizes[c] < 30: break
    C = lab == (c + 1); Cd = ndimage.binary_dilation(C, S26, iterations=1)
    k = genus_solid(mesh_solid & ~Cd) - g_mesh_vox
    shell = ndimage.binary_dilation(C, S26, iterations=1) & ~C
    mouths = shell & mesh_air & ~hs
    lm, nm = ndimage.label(mouths, structure=S26); ms = np.bincount(lm.ravel())[1:] if nm else np.zeros(0, int)
    mlist = [np.argwhere(lm == i + 1) for i in np.argsort(-ms) if ms[i] >= 8]
    comps.append((C, k, mlist)); print(f"    comp {int(sizes[c]):6d} vox  k={k:+d}  mouths={[len(m) for m in mlist]}  centre={np.round(v2w(np.argwhere(C).mean(0)),3)}")
# ---- render: 2 rows x 3 views. Row 1: hull (grey) + blocks coloured by k. Row 2: same + mouths (one colour per mouth).
hv, hf, _, _ = measure.marching_cubes(hs.astype(np.float32), 0.5); hv = v2w(hv)
ext = hv.max(0) - hv.min(0); c0 = (hv.max(0) + hv.min(0)) / 2
fig = plt.figure(figsize=(21, 12)); fig.patch.set_facecolor("white")
cols_k = {0: "#bbbbbb", 1: "#e6a23c", 2: "#d9534f", 3: "#8e44ad"}; mouth_cols = ["#1f77b4", "#2ca02c", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
views3 = [(25, -60), (25, 30), (60, 120)]
for row in range(2):
    for vi, (el, az) in enumerate(views3):
        ax = fig.add_subplot(2, 3, row * 3 + vi + 1, projection="3d")
        ax.scatter(hv[::5, 0], hv[::5, 1], hv[::5, 2], s=1, c="#c8c8c8", alpha=0.12, depthshade=False)
        for ci, (C, k, mlist) in enumerate(comps):
            p = v2w(np.argwhere(C))
            if row == 0 or k <= 0:
                ax.scatter(p[::2, 0], p[::2, 1], p[::2, 2], s=5, c=cols_k.get(max(k, 0), "#000"), alpha=0.9 if k > 0 else 0.2, depthshade=False)
                if k > 0: cen = p.mean(0); ax.text(cen[0], cen[1], cen[2], f"k={k}", fontsize=11, weight="bold")
            else:
                ax.scatter(p[::4, 0], p[::4, 1], p[::4, 2], s=2, c=cols_k.get(k, "#000"), alpha=0.12, depthshade=False)
                for mi, m in enumerate(mlist[:6]):
                    q = v2w(m); ax.scatter(q[:, 0], q[:, 1], q[:, 2], s=12, c=mouth_cols[mi % 6], depthshade=False)
                    qc = q.mean(0); ax.text(qc[0], qc[1], qc[2], f"m{mi+1}", fontsize=9, color=mouth_cols[mi % 6], weight="bold")
        ax.set_xlim(c0[0]-ext[0]/2, c0[0]+ext[0]/2); ax.set_ylim(c0[1]-ext[1]/2, c0[1]+ext[1]/2); ax.set_zlim(c0[2]-ext[2]/2, c0[2]+ext[2]/2)
        ax.view_init(el, az); ax.set_axis_off(); ax.set_box_aspect(tuple(ext / ext.max()))
        ax.set_title(("blocks (colour = k)" if row == 0 else "mouths (block surface touching mesh air)") + f"  view {vi+1}", fontsize=10)
fig.suptitle(f"{shape} {os.path.basename(npz)} | mesh genus {g_mesh}, hull genus {genus_solid(hs)} | mesh material where the hull says air (d={D}): orange k=1, red k=2, purple k=3, grey k=0 (fitting slop)", fontsize=11)
fig.tight_layout(); fig.savefig(out, dpi=100); print("saved", out)
