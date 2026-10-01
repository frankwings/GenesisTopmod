"""Per-handle GT audit from the per-candidate site log (works for every shape, drills and joins).
For each ACCEPTED handle: sample GT occupancy along the segment between the two face centroids.
  drill (MEMBRANE, or INVALID kept by render evidence) is right iff GT is AIR there;
  join  (CONTACT / CREASE, or overlapping parts)       is right iff GT is MATERIAL there.
Usage: SHAPE=fertility python3 despike/audit_sitelog_gt.py <tag_prefix> <log_file> <log_label>
  e.g. SHAPE=fertility python3 despike/audit_sitelog_gt.py g67f v67_fert.log V67F"""
import os, sys, re, json, glob, numpy as np
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5"); sys.path.insert(0, "despike"); sys.path.insert(0, ".")
from viz_render import load
import open3d as o3d
SHAPE = os.environ["SHAPE"]; prefix, logf, lab = sys.argv[1], sys.argv[2], sys.argv[3]
R = "despike/results_genus"; LOG = open(logf if os.path.exists(logf) else f"/home/kingy/Foundation/EdenGateway/agents/hani/bg_tasks/logs/{logf}").read().splitlines()
tags = sorted({m.group(1) for l in LOG for m in [re.search(rf"\[{lab} ({prefix}\w*)\] \[RESULT\] {SHAPE} ", l)] if m}, key=lambda t: (len(t), t))
Vg, Fg = load("GT"); Vg = np.asarray(Vg, float); Fg = np.asarray(Fg, np.int64)
ref = np.load(os.environ.get("REF_NPZ", f"{R}/{SHAPE}_{tags[0]}_auto.npz"))["verts"].astype(float); T0 = np.eye(4); T0[:3, 3] = ref.mean(0) - Vg.mean(0)
reg = o3d.pipelines.registration.registration_icp(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Vg)), o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref)), 0.05, T0, o3d.pipelines.registration.TransformationEstimationPointToPoint())
Vg = (np.c_[Vg, np.ones(len(Vg))] @ reg.transformation.T)[:, :3]
sc = o3d.t.geometry.RaycastingScene(); sc.add_triangles(o3d.core.Tensor(Vg.astype(np.float32)), o3d.core.Tensor(Fg.astype(np.uint32)))
def gt_mat(ci, cj, n=9):
    ts = np.linspace(0.15, 0.85, n); S = (np.asarray(ci)[None] * (1 - ts[:, None]) + np.asarray(cj)[None] * ts[:, None]).astype(np.float32)
    return float((sc.compute_occupancy(o3d.core.Tensor(S)).numpy() > 0.5).mean())
tot = ok = 0; flagged = []; nomatch = 0
for t in tags:
    gates = [('"gated_accept": true' in l) for l in LOG if l.startswith(f"[{lab} {t}] [{SHAPE} ") and "JEV gate" in l]
    sites = []
    for f in (f"{R}/sitelog_{SHAPE}_{t}.jsonl", f"{R}/sitelog_{SHAPE}_{t}_5b.jsonl"):
        if os.path.exists(f): sites += [json.loads(x) for x in open(f) if x.strip()]
    batch = [x for x in sites if "decision" in x]; sites = [x for x in sites if "decision" not in x]   # BATCH_OPEN entries carry their own decision
    if len(gates) != len(sites): nomatch += 1; continue
    for acc, s in [(x["decision"] == "batch_accept", x) for x in batch] + list(zip(gates, sites)):
        if not acc: continue
        join = s["site"] in ("CONTACT", "CREASE") or (s.get("overlap") or 0) >= 0.6
        g = gt_mat(s["ci"], s["cj"]); good = (g >= 0.6) if join else (g <= 0.4)
        tot += 1; ok += good
        if not good: flagged.append((t, s["site"], "join" if join else "drill", round(g, 2), s.get("w_med"), round(s["air"], 2), round(s["geo_ratio"])))
print(f"{SHAPE}: runs={len(tags)} (log/sitelog count mismatch in {nomatch}), accepted handles audited={tot}, consistent with GT={ok}, flagged={len(flagged)}  (GT icp fitness {reg.fitness:.3f})")
for x in flagged: print("   FLAG", x)
