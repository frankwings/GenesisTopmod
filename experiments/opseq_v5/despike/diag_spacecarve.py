"""Space-carving diagnostic (Boss's visual-hull idea, 2026-09-01).

Evidence chain:
  red pixel in view i (pred=1, GT=0)  =>  no material anywhere on that ray.
  64-view GT silhouette intersection  =>  GT visual hull.
  excess_hull = our_mesh_occupancy AND (NOT gt_hull)
             => material PROVABLY removable from silhouettes alone
                (no GT-mesh oracle; legitimate supervision for the method).

Also computed for reference (oracle, diagnosis only):
  excess_true  = mesh_occ AND NOT gt_occ
  missing_true = gt_occ  AND NOT mesh_occ

Per-component analysis of excess_hull: size, bbox, and exterior-contact patch
count (>=2 locally-disconnected air contacts => through-cut needed: slit if the
contacts join around a rim nearby, hole otherwise).

Run:  python3 despike/diag_spacecarve.py   (from experiments/opseq_v5)
Outputs: /tmp/liou_cow_viz/diag_spacecarve.{png,npz} + stdout report.
"""
import sys, os
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")

import numpy as np, torch
import open3d as o3d
from scipy.ndimage import label as cc_label, binary_dilation
import nvdiffrast.torch as dr
import run_64v
from run_64v import star_cameras, make_gt, NV
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH, IMG_RES
from cow_v13 import DEVICE

SHAPE = os.environ.get("SHAPE", "armadillo")
NPZ = os.environ.get("NPZ", "/tmp/liou_cow_viz/cow_armadillo_p1d64c.npz")
NRES = int(os.environ.get("NRES", "192"))
OUTD = "/tmp/liou_cow_viz"

# ---------------------------------------------------------------- scene setup
ctx = dr.RasterizeCudaContext()
gv, gf = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj"))
gv = normalize_to_range(gv)
max_r = float(np.linalg.norm(gv, axis=1).max())
mvps, views = star_cameras(max_r)
gt_sils, _, _, _ = make_gt(ctx, mvps, views, SHAPE)          # [64,H,W] u8, inside<128

d = np.load(NPZ)
mv, mf = d["verts"].astype(np.float64), d["tris"].astype(np.int64)
print(f"[diag] shape={SHAPE} mesh V={len(mv)} F={len(mf)} NRES={NRES}", flush=True)

# ---------------------------------------------------------------- voxel grid
lo = np.minimum(gv.min(0), mv.min(0)) - 0.02
hi = np.maximum(gv.max(0), mv.max(0)) + 0.02
axes = [np.linspace(lo[a], hi[a], NRES) for a in range(3)]
pitch = float((hi - lo).max() / (NRES - 1))
G = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)  # [M,3]
M = G.shape[0]

def occupancy(V, F):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(
        o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32))))
    out = np.zeros(M, bool)
    CH = 2_000_000
    for s in range(0, M, CH):
        q = o3d.core.Tensor(G[s:s+CH].astype(np.float32))
        out[s:s+CH] = sc.compute_occupancy(q).numpy() > 0.5
    return out.reshape(NRES, NRES, NRES)

gt_occ = occupancy(gv, gf)
mesh_occ = occupancy(mv, mf)
print(f"[diag] gt_occ={gt_occ.sum()} mesh_occ={mesh_occ.sum()} voxels "
      f"(pitch={pitch:.4f})", flush=True)

# ---------------------------------------------------------------- GT visual hull
sil_in = torch.from_numpy((gt_sils < 128)).to(DEVICE)        # [64,H,W] bool inside
Gt = torch.from_numpy(G.astype(np.float32)).to(DEVICE)
ones = torch.ones(M, 1, device=DEVICE)
Gh = torch.cat([Gt, ones], 1)                                # [M,4]
hull = torch.ones(M, dtype=torch.bool, device=DEVICE)
H = W = IMG_RES
for i in range(NV):
    clip = (mvps[i] @ Gh.T).T                                # [M,4]
    w = clip[:, 3].clamp(min=1e-8)
    x, y = clip[:, 0] / w, clip[:, 1] / w
    u = ((x + 1) * 0.5 * W).long().clamp(0, W - 1)
    v = ((y + 1) * 0.5 * H).long().clamp(0, H - 1)
    inb = (x.abs() <= 1) & (y.abs() <= 1)
    hull &= inb & sil_in[i][v, u]
gt_hull = hull.cpu().numpy().reshape(NRES, NRES, NRES)
print(f"[diag] gt_hull={gt_hull.sum()} voxels "
      f"(hull/gt_occ ratio={gt_hull.sum()/max(gt_occ.sum(),1):.3f})", flush=True)

# ---------------------------------------------------------------- evidence sets
excess_hull = mesh_occ & ~gt_hull      # silhouette-provable removals (actionable)
excess_true = mesh_occ & ~gt_occ       # oracle
missing_true = gt_occ & ~mesh_occ      # oracle
vol = lambda m: m.sum()
print(f"\n[diag] mesh volume          : {vol(mesh_occ)}")
print(f"[diag] excess_hull (provable): {vol(excess_hull)}  "
      f"({100*vol(excess_hull)/vol(mesh_occ):.2f}% of mesh)")
print(f"[diag] excess_true (oracle)  : {vol(excess_true)}  "
      f"({100*vol(excess_true)/vol(mesh_occ):.2f}%)")
print(f"[diag] provable/true excess  : "
      f"{100*vol(excess_hull)/max(vol(excess_true),1):.1f}%")
print(f"[diag] missing_true (oracle) : {vol(missing_true)}", flush=True)

# ---------------------------------------------------------------- components
lab, ncc = cc_label(excess_hull, structure=np.ones((3, 3, 3), int))
sizes = np.bincount(lab.ravel())[1:]
order = np.argsort(sizes)[::-1]
ext_air = ~mesh_occ
print(f"\n[diag] excess_hull components: {ncc}  (top 15 below)")
rows = []
for r, ci in enumerate(order[:15]):
    comp = lab == (ci + 1)
    idx = np.argwhere(comp)
    bb0, bb1 = idx.min(0), idx.max(0)
    dil = binary_dilation(comp, iterations=2)
    contact = dil & ext_air
    # local air-patch count inside comp bbox (+4 margin)
    m0 = np.maximum(bb0 - 4, 0); m1 = np.minimum(bb1 + 5, NRES)
    sub = contact[m0[0]:m1[0], m0[1]:m1[1], m0[2]:m1[2]]
    _, npatch = cc_label(sub, structure=np.ones((3, 3, 3), int))
    kind = "THROUGH(slit/hole)" if npatch >= 2 else "pocket"
    ext_wd = (bb1 - bb0 + 1)
    print(f"  #{r:2d} size={sizes[ci]:6d}  extent={tuple(ext_wd)}  "
          f"air_patches={npatch}  -> {kind}")
    rows.append((int(sizes[ci]), npatch))

n_through = sum(1 for _, p in rows if p >= 2)
print(f"\n[diag] top-15: {n_through} THROUGH-type vs {15 - n_through} pocket-type")

np.savez_compressed(os.path.join(OUTD, "diag_spacecarve.npz"),
                    lo=lo, hi=hi, nres=NRES,
                    mesh_occ=mesh_occ, gt_hull=gt_hull, gt_occ=gt_occ,
                    excess_hull=excess_hull)

# ---------------------------------------------------------------- viz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig = plt.figure(figsize=(16, 10))
# 3D scatter: mesh surface (grey) + excess_hull (red)
ax = fig.add_subplot(2, 3, 1, projection="3d")
surf = mesh_occ & ~binary_dilation(~mesh_occ, iterations=1) == False  # noqa
shell = mesh_occ & binary_dilation(~mesh_occ)
si = np.argwhere(shell); ei = np.argwhere(excess_hull)
ss = np.random.choice(len(si), min(9000, len(si)), replace=False)
ax.scatter(si[ss, 0], si[ss, 1], si[ss, 2], s=1, c="lightgrey", alpha=0.15)
if len(ei):
    es = np.random.choice(len(ei), min(20000, len(ei)), replace=False)
    ax.scatter(ei[es, 0], ei[es, 1], ei[es, 2], s=2, c="red", alpha=0.6)
ax.set_title(f"excess_hull ({vol(excess_hull)} vox) on mesh shell")
ax.view_init(elev=12, azim=60); ax.set_box_aspect((1, 1, 1)); ax.axis("off")
ax2 = fig.add_subplot(2, 3, 2, projection="3d")
ax2.scatter(si[ss, 0], si[ss, 1], si[ss, 2], s=1, c="lightgrey", alpha=0.15)
if len(ei):
    ax2.scatter(ei[es, 0], ei[es, 1], ei[es, 2], s=2, c="red", alpha=0.6)
ax2.set_title("azim=150")
ax2.view_init(elev=12, azim=150); ax2.set_box_aspect((1, 1, 1)); ax2.axis("off")
# missing (green) for context
ax3 = fig.add_subplot(2, 3, 3, projection="3d")
mi = np.argwhere(missing_true)
ax3.scatter(si[ss, 0], si[ss, 1], si[ss, 2], s=1, c="lightgrey", alpha=0.15)
if len(mi):
    ms = np.random.choice(len(mi), min(20000, len(mi)), replace=False)
    ax3.scatter(mi[ms, 0], mi[ms, 1], mi[ms, 2], s=2, c="green", alpha=0.6)
ax3.set_title(f"missing_true oracle ({vol(missing_true)} vox)")
ax3.view_init(elev=12, azim=60); ax3.set_box_aspect((1, 1, 1)); ax3.axis("off")
# slices through the largest excess component
if len(order):
    comp = lab == (order[0] + 1)
    cz = np.argwhere(comp)[:, 2]
    zs = np.percentile(cz, [25, 50, 75]).astype(int)
    for j, z in enumerate(zs):
        axs = fig.add_subplot(2, 3, 4 + j)
        img = np.zeros((NRES, NRES, 3))
        img[mesh_occ[:, :, z]] = [0.75, 0.75, 0.75]
        img[gt_hull[:, :, z] & mesh_occ[:, :, z]] = [0.55, 0.55, 0.55]
        img[excess_hull[:, :, z]] = [1, 0.2, 0.2]
        img[comp[:, :, z]] = [1, 0, 0]
        axs.imshow(np.rot90(img)); axs.set_title(f"slice z={z} (grey=mesh, red=excess)")
        axs.axis("off")
fig.tight_layout()
fig.savefig(os.path.join(OUTD, "diag_spacecarve.png"), dpi=110)
print(f"[diag] wrote {OUTD}/diag_spacecarve.png + .npz", flush=True)
