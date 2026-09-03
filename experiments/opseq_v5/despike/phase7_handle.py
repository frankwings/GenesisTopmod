"""Phase 7: genus change from space-carving evidence + DLFL add_handle.

A hole the sphere-init chain cannot make shows up as a dimple: two patches of
our surface pushed together inside the GT tunnel. Both patches lie OUTSIDE the
voting visual hull (the hull has a tunnel there), their outward normals face
each other, and the segment between them stays outside the hull. That is
proof of a tunnel. The two faces are back-to-back (normals away from each
other, our slab between them). DLFL add_handle(face_i, face_j): both faces removed, a
tube of side quads inserted through the slab (genus +1, manifold preserved), quads stellated.
Then the normal loop (phase4) pulls the tube walls to the hole wall.

Run: MODE=64v SHAPE=rockerarm BASE_NPZ=... TAG=... [MAX_HANDLES=1] python3 despike/phase7_handle.py
"""
import sys, os, tempfile
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod")
os.chdir("/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
os.environ.setdefault("MODE", "64v")
import numpy as np, torch
import nvdiffrast.torch as dr
import cow_v13
from cow_v13 import DEVICE
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
import phase1b_pipeline as p1b
from phase1b_pipeline import heldout_exam, check_watertight
from hull_field import build_vote_hull
from topmod.io import from_obj, to_triangle_arrays
from topmod.high_level_ops import add_handle, stellate as dlfl_stellate
import run_64v

SHAPE = os.environ.get("SHAPE", "rockerarm")
TAG = os.environ.get("TAG", f"{SHAPE}_p7")
BASE_NPZ = os.environ["BASE_NPZ"]
MAX_HANDLES = int(os.environ.get("MAX_HANDLES", "1"))
OUT_VOX = float(os.environ.get("OUT_VOX", "2.0"))     # both faces must be > this many voxels outside the hull
FACE_COS = float(os.environ.get("FACE_COS", "-0.5"))  # n_i . n_j below this (facing each other)
MAX_SEP = float(os.environ.get("MAX_SEP", "6.0"))     # max centroid separation in mean-edge units
OUTD = "/tmp/liou_cow_viz"

z = np.load(BASE_NPZ); V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ctx = dr.RasterizeCudaContext()
gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj")); gvn = normalize_to_range(gv)
mvps, views = run_64v.star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
gt, gtd, gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE); cow_v13.N_VIEWS = 64
p1b._MVPS, p1b._GT = mvps, gt; p1b.SHAPE = SHAPE
HF = build_vote_hull(ctx, mvps, gvn, gf_gt, V, DEVICE, nres=256, hires=512, vote=2)
pitch = HF.pitch

def genus(V, F):
    E = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0))
    return (2 - (len(V) - E + len(F))) // 2

def hdist(P):
    return HF.dist(torch.tensor(np.asarray(P, np.float32), device=DEVICE)).detach().cpu().numpy()

def report(tag, V, F):
    ho = heldout_exam(ctx, V, F); wt, _ = check_watertight(F)
    print(f"[{tag}] V={len(V)} F={len(F)} watertight={wt} genus={genus(V, F)} | ho16={ho[0]:.4f} hair={ho[1]} maxblob={ho[2]}", flush=True)

def find_tunnel_pairs(V, F):
    tri = V[F]; cen = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); me = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()
    d = hdist(cen)
    cand = np.where(d > OUT_VOX * pitch)[0]
    print(f"[p7] faces outside hull by >{OUT_VOX} voxels: {len(cand)} / {len(F)} (pitch {pitch:.4f}, mean edge {me:.4f})", flush=True)
    pairs = []
    for a in range(len(cand)):
        i = cand[a]
        for j in cand[a + 1:]:
            if len(set(F[i]) & set(F[j])): continue                   # must not share vertices
            v = cen[j] - cen[i]; L = np.linalg.norm(v)
            if L > MAX_SEP * me or L < 1e-6: continue
            if n[i] @ n[j] > FACE_COS: continue                          # facing each other
            # membrane = thin slab of OUR volume inside the GT tunnel: the two faces are
            # back-to-back, outward normals point AWAY from each other
            if n[i] @ v >= 0 or n[j] @ (-v) >= 0: continue
            seg = cen[i] + np.linspace(0.1, 0.9, 9)[:, None] * v
            if (hdist(seg) < 0.5 * pitch).any(): continue               # whole segment outside hull
            pairs.append((min(d[i], d[j]) / pitch, -L / me, int(i), int(j)))
    pairs.sort(reverse=True)
    return pairs, cen, d

report("base", V, Fa)
n_added = 0
for k in range(MAX_HANDLES):
    pairs, cen, d = find_tunnel_pairs(V, Fa)
    print(f"[p7] tunnel-evidence pairs: {len(pairs)}", flush=True)
    if not pairs: break
    score, negL, i, j = pairs[0]
    print(f"[p7] add_handle between faces {i},{j}: out {d[i]/pitch:.1f}/{d[j]/pitch:.1f} vox, sep {-negL:.2f} edges, centroids {np.round(cen[i],3)} {np.round(cen[j],3)}", flush=True)
    with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
        for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
        for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
        path = fh.name
    mesh = from_obj(path); os.unlink(path)
    faces = list(mesh.iter_faces())
    add_handle(mesh, faces[i], faces[j])
    for f in list(mesh.faces.values()):
        if len(f.vertices()) > 3: dlfl_stellate(mesh, f)
    vv, ff = to_triangle_arrays(mesh)
    V2, F2 = np.asarray(vv, float), np.asarray(ff, np.int64)
    wt, nbad = check_watertight(F2); assert wt, nbad
    assert np.allclose(V2[:len(V)], V, atol=1e-9)
    V, Fa = V2, F2; n_added += 1
    report(f"after handle {n_added}", V, Fa)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
print(f"[p7] handles added: {n_added}; saved cow_{SHAPE}_{TAG}.npz", flush=True)
