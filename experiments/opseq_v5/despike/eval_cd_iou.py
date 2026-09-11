"""Chamfer Distance + Volume IoU after ICP, the metric family of the 3DV-2026 / ICASSP-2026
high-genus papers (they do not publish sample counts / voxel resolution; ours: ICP point-to-plane
on 50k samples, CD = mean of both nearest-neighbour directions over 100k samples each,
Volume IoU on a 256^3 occupancy grid over the GT bounding box, both meshes watertight).
Usage: SHAPE=fertility python3 despike/eval_cd_iou.py label=path.npz [label=path ...]"""
import sys, os, numpy as np
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
import open3d as o3d
from viz_render import load
GRID = int(os.environ.get("GRID", "256")); NS = int(os.environ.get("NS", "100000"))

def o3dmesh(V, F):
    m = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(np.asarray(V, float)), o3d.utility.Vector3iVector(np.asarray(F, np.int32)))
    m.compute_vertex_normals(); return m

def icp_align(V, F, Vg, Fg):
    src = o3dmesh(V, F).sample_points_uniformly(50000); dst = o3dmesh(Vg, Fg).sample_points_uniformly(50000)
    src.estimate_normals(); dst.estimate_normals()
    reg = o3d.pipelines.registration.registration_icp(src, dst, 0.05, np.eye(4),
          o3d.pipelines.registration.TransformationEstimationPointToPlane(),
          o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100))
    T = reg.transformation; Vh = np.c_[V, np.ones(len(V))] @ T.T
    return Vh[:, :3], float(reg.fitness)

def chamfer(V, F, Vg, Fg):
    a = np.asarray(o3dmesh(V, F).sample_points_uniformly(NS).points); b = np.asarray(o3dmesh(Vg, Fg).sample_points_uniformly(NS).points)
    ka = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a)))
    kb = o3d.geometry.KDTreeFlann(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b)))
    def nn(pts, tree): return np.array([np.sqrt(tree.search_knn_vector_3d(p, 1)[2][0]) for p in pts])
    return float(0.5 * (nn(a, kb).mean() + nn(b, ka).mean()))

def occupancy(V, F, lo, hi):
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.int32))))
    g = [np.linspace(lo[i], hi[i], GRID, dtype=np.float32) for i in range(3)]
    P = np.stack(np.meshgrid(*g, indexing="ij"), -1).reshape(-1, 3)
    occ = np.zeros(len(P), bool)
    for s in range(0, len(P), 2_000_000):
        occ[s:s+2_000_000] = sc.compute_occupancy(o3d.core.Tensor(P[s:s+2_000_000])).numpy() > 0.5
    return occ

if __name__ == "__main__":
    Vg, Fg = load("GT"); Vg = np.asarray(Vg, float); Fg = np.asarray(Fg, np.int64)
    lo, hi = Vg.min(0) - 0.05, Vg.max(0) + 0.05
    og = occupancy(Vg, Fg, lo, hi)
    for arg in sys.argv[1:]:
        lab, p = arg.split("=", 1); V, F = load(p); V = np.asarray(V, float); F = np.asarray(F, np.int64)
        Va, fit = icp_align(V, F, Vg, Fg)
        cd = chamfer(Va, F, Vg, Fg); o = occupancy(Va, F, lo, hi)
        iou = float((o & og).sum() / max((o | og).sum(), 1))
        print(f"[cd_iou] {lab:28s} V={len(V):6d} | CD={cd:.5f}  VolIoU={iou:.4f}  (icp fitness {fit:.3f}) | GT bbox-normalised CD={cd/np.linalg.norm(hi-lo):.5f}", flush=True)
