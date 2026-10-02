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
TUBE_SUBDIV = int(os.environ.get("TUBE_SUBDIV", "0"))
PROJECT = int(os.environ.get("PROJECT", "0"))
EXT_VOX = float(os.environ.get("EXT_VOX", "8.0"))
MIN_SEP = float(os.environ.get("MIN_SEP", "1.5"))
PROJ_RADIUS = float(os.environ.get("PROJ_RADIUS", "0.6"))  # also radially project the membrane around the tube (within this radius of the axis) so the mouth eats it; 0 = tube only    # faces closer than this (mean-edge units) are fold/sliver remnants, not a slab   # tunnel must continue hull-free this far beyond BOTH faces (bay vs through-hole)
DRY = int(os.environ.get("DRY", "0"))
PRE_SUBDIV = int(os.environ.get("PRE_SUBDIV", "0"))   # 1 = always pre-subdivide entry/exit faces before add_handle (auto when their 1-rings overlap)
DETECT = os.environ.get("DETECT", "membrane")   # membrane (Boss 2026-09-16: DR vertices outside the hull) | hull (closing ladder) | rays
ABSORB = int(os.environ.get("ABSORB", "0"))
OPEN = os.environ.get("OPEN", "merge")   # merge = collapse membrane interior verts to the rim, delete interior edges -> rim polygon, add_handle(rim1, rim2)   # after add_handle, eat the blocking membrane into the mouth by DLFL collapses (mouth ring grows to the membrane rim)
HANDLES_JSON = os.environ.get("HANDLES_JSON", "")     # persisted list of handle midpoints across rounds (one handle per tunnel)
R_DEDUP = float(os.environ.get("R_DEDUP", "0.3"))      # a new candidate closer than this to an existing handle is the SAME tunnel -> skip   # rays = see-through pixels of the training silhouettes (default) | hull = back-to-back faces outside the hull             # 1 = detect and report only       # after refinement, project every vertex that is outside the hull onto the hull boundary (deterministic inflate)  # DLFL-subdivide the new tube faces N times so the hull field can inflate a long tube
OUT_VOX = float(os.environ.get("OUT_VOX", "4.0"))     # both faces must be > this many voxels outside the hull
FACE_COS = float(os.environ.get("FACE_COS", "-0.5"))  # n_i . n_j below this (facing each other)
MAX_SEP = float(os.environ.get("MAX_SEP", "100.0"))   # max centroid separation (mean-edge units); thick slabs need long tubes (3holes: 12 edges)
OUTD = "/tmp/liou_cow_viz"
HULL_LOC_RES = int(os.environ.get("HULL_LOC_RES", "128"))
HULL_FACE_DIST_VOX = float(os.environ.get("HULL_FACE_DIST_VOX", "6"))
HULL_PLUGS_CACHE = os.environ.get("HULL_PLUGS_CACHE", f"{OUTD}/hull_plugs_{SHAPE}_{HULL_LOC_RES}.json")

z = np.load(BASE_NPZ); V, Fa = z["verts"].astype(np.float64), z["tris"].astype(np.int64)
ctx = dr.RasterizeCudaContext()
REAL_DATA = os.environ.get("REAL_DATA", "")
if REAL_DATA:
    from real_scene import load_real_scene
    _real_scene = load_real_scene(REAL_DATA, DEVICE)
    mvps, views = _real_scene.mvps, _real_scene.views
    gt = _real_scene.gt
    HF = _real_scene.hull(ctx, extra_pts=V)
    p1b.heldout_exam = _real_scene.heldout_exam
    heldout_exam = _real_scene.heldout_exam   # rebind the name imported at module top
else:
    gv, gf_gt = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), f"{SHAPE}.obj")); gvn = normalize_to_range(gv)
    mvps, views = run_64v.star_cameras(float(np.linalg.norm(gvn, axis=1).max()))
    gt, gtd, gtdiff, _ = run_64v.make_gt(ctx, mvps, views, SHAPE)
    HF = build_vote_hull(ctx, mvps, gvn, gf_gt, None, DEVICE, nres=256, hires=int(os.environ.get("HULL_HIRES", "512")), vote=2)   # GT-bbox grid (not widened by the current mesh): plug detection must not depend on run-to-run mesh noise
    SITE_V2 = int(os.environ.get("SITE_V2", "1"))
    _AIR_MASKS = None
    if SITE_V2:
        # Exact hull-air test for individual points (2026-09-29, user suggestion): instead of sampling the
        # 256^3 voxel hull with a half-voxel margin, project each query point into all training
        # silhouettes directly (the carving rule itself, at image resolution = an infinitely refined octree).
        from pipeline.cameras import transform_to_clip
        import torch.nn.functional as _Fnn
        _AIR_RES = int(os.environ.get("AIR_RES", "1024"))
        _gvt = torch.tensor(gvn, dtype=torch.float32, device=DEVICE); _gft = torch.tensor(np.asarray(gf_gt, np.int32), dtype=torch.int32, device=DEVICE)
        _AIR_MASKS = []
        with torch.no_grad():
            for _k in range(len(mvps)):
                _r, _ = dr.rasterize(ctx, transform_to_clip(_gvt, mvps[_k]), _gft, resolution=[2 * _AIR_RES, 2 * _AIR_RES])
                _cov = _Fnn.avg_pool2d((_r[0, :, :, 3] > 0).float()[None, None], 2)[0, 0]
                _AIR_MASKS.append(_cov >= 0.25)          # unbiased edge (hull_field ss_thr mode)
        print(f"[p7] SITE_V2: exact silhouette air test on {len(_AIR_MASKS)} views at {_AIR_RES}px", flush=True)
cow_v13.N_VIEWS = 64
SITE_V2 = globals().get("SITE_V2", int(os.environ.get("SITE_V2", "1"))); _AIR_MASKS = globals().get("_AIR_MASKS", None)
p1b._MVPS, p1b._GT = mvps, gt; p1b.SHAPE = SHAPE
pitch = HF.pitch
from hull_field import hull_genus
_GT_ENV = os.environ.get("GENUS_TARGET", "hull")   # hull = persistent genus of the space-carved hull (LESSONS 24) | off | integer
if _GT_ENV == "off": G_TARGET = None
elif _GT_ENV == "hull": G_TARGET, _gr = hull_genus(HF.hull); print(f"[p7] genus target from space-carved hull: g*={G_TARGET} (per radius {_gr})", flush=True)
else: G_TARGET = int(_GT_ENV)
RELAX = [(float(os.environ.get("MIN_PX", "30")), OUT_VOX), (15.0, 2.0), (8.0, 1.0)]   # (min_px, out_vox) ladder while genus < g*

def genus(V, F):
    E = len(np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1), axis=0))
    return (2 - (len(V) - E + len(F))) // 2

def hdist(P):
    return HF.dist(torch.tensor(np.asarray(P, np.float32), device=DEVICE)).detach().cpu().numpy()

def hull_air_exact(P, vote=2):
    """Point is hull AIR iff >= `vote` training silhouettes see background at its projection (same rule as
    carve_hull, evaluated per point at image resolution; no voxels, no margin). Falls back to the voxel
    distance field (margin 0) when no masks are available (real-data path)."""
    if _AIR_MASKS is None:
        return hdist(P) > 0
    Pt = torch.tensor(np.asarray(P, np.float32), device=DEVICE); Ph = torch.cat([Pt, torch.ones(len(Pt), 1, device=DEVICE)], 1)
    votes = torch.zeros(len(Pt), dtype=torch.int32, device=DEVICE); R = _AIR_MASKS[0].shape[0]
    for k, m in enumerate(_AIR_MASKS):
        mvp = mvps[k] if isinstance(mvps[k], torch.Tensor) else torch.tensor(mvps[k], dtype=torch.float32, device=DEVICE)
        c = (mvp.to(DEVICE).float() @ Ph.T).T; w = c[:, 3].clamp(min=1e-8); x = c[:, 0] / w; y = c[:, 1] / w
        inb = (x.abs() <= 1) & (y.abs() <= 1)
        ui = ((x + 1) * 0.5 * R).long().clamp(0, R - 1); vi = ((y + 1) * 0.5 * R).long().clamp(0, R - 1)
        votes += (inb & ~m[vi, ui]).int()
    return (votes >= vote).cpu().numpy()

def normal_side(V, F, fi, fj, tol=0.2):
    """User suggestion 2026-09-29: decide inside/outside of the segment between two faces from their
    normals alone. Back-to-back (each face's partner lies BEHIND it) -> the segment is inside our solid
    (membrane pages); face-to-face (partner IN FRONT) -> outside (contact gap). Returns 1.0 / 0.0, or None
    when the two faces disagree or are too oblique (then fall back to ray parity). Orientation is taken
    from the mesh's signed volume so it works whether faces are wound outward or inward."""
    V = np.asarray(V, float); F = np.asarray(F, np.int64)
    tri = V[F]; vol = float(np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum())
    sgn = 1.0 if vol >= 0 else -1.0
    def nrm(f):
        t = V[F[int(f)]]; n = np.cross(t[1] - t[0], t[2] - t[0]); return sgn * n / (np.linalg.norm(n) + 1e-12), t.mean(0)
    ni, ci = nrm(fi); nj, cj = nrm(fj); d = cj - ci; L = np.linalg.norm(d) + 1e-12
    si = float(ni @ d) / L; sj = float(nj @ (-d)) / L
    if si < -tol and sj < -tol: return 1.0
    if si > tol and sj > tol: return 0.0
    return None

def winding_numbers(V, F, P):
    """Generalized winding number (Jacobson et al. 2013) of our mesh at points P: 0 outside, 1 inside,
    2 where two parts of the surface interpenetrate (e.g. an arm pushed into the body). Unlike ray parity it
    is robust to self-intersections, and unlike a local normal test it COUNTS how many volumes contain P."""
    V = np.asarray(V, float); F = np.asarray(F, np.int64); P = np.asarray(P, float)
    tri = V[F]
    if np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() < 0: F = F[:, [0, 2, 1]]
    out = np.zeros(len(P))
    for s in range(0, len(F), 20000):
        Fb = F[s:s + 20000]
        A = V[Fb[:, 0]][None] - P[:, None]; B = V[Fb[:, 1]][None] - P[:, None]; C = V[Fb[:, 2]][None] - P[:, None]
        a = np.linalg.norm(A, axis=2); b = np.linalg.norm(B, axis=2); c = np.linalg.norm(C, axis=2)
        num = np.einsum("pfi,pfi->pf", A, np.cross(B, C))
        den = a * b * c + np.einsum("pfi,pfi->pf", A, B) * c + np.einsum("pfi,pfi->pf", B, C) * a + np.einsum("pfi,pfi->pf", C, A) * b
        out += (2 * np.arctan2(num, den)).sum(1)
    return out / (4 * np.pi)

def membrane_check(V, F, ci, cj, n=7, fi=None, fj=None):
    """Is the stretch between the two faces a genuine MEMBRANE = our material lying in hull air?
    inside = fraction of interior samples inside the current mesh (material between the two pages)
    air    = fraction of interior samples outside the carved hull (space the silhouettes prove empty)
    Only a membrane passes both. An already-open tunnel (air, no material) fails `inside` -> add_handle
    would build a BRIDGE; a real solid slab such as a base plate (material, hull solid) fails `air`.
    (2026-09-27 fix: hull-guided completion bridged the open base arch while genus still read 4.)"""
    import open3d as o3d
    ts = np.linspace(0.15, 0.85, n)
    S = (np.asarray(ci)[None, :] * (1 - ts[:, None]) + np.asarray(cj)[None, :] * ts[:, None]).astype(np.float32)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32)))
    inside_par = float((sc.compute_occupancy(o3d.core.Tensor(S)).numpy() > 0.5).mean())
    if SITE_V2:
        # 2026-09-29: the generalized winding number is the primary inside measure (it also detects
        # interpenetration, w~2, which a local normal test and ray parity both miss); normals = cross-check.
        w = winding_numbers(V, F, S)
        global _LAST_OVERLAP
        # 2026-09-30: use |w|. w = -1 is a membrane whose two pages were pushed THROUGH each other by DR
        # (self-collision, "negative thickness"): still surplus material blocking the tunnel (14/14 such
        # candidates in the v66 regression were real tunnels per GT). |w| >= 1.5 = two parts interpenetrate.
        aw = np.abs(w)
        _LAST_OVERLAP = float((aw >= 1.5).mean())
        global _LAST_WMED
        _LAST_WMED = float(np.median(w))
        inside = float((aw >= 0.5).mean())
        air = float(np.asarray(hull_air_exact(S)).mean())
        ns = normal_side(V, F, fi, fj) if (fi is not None and fj is not None) else None
        print(f"[p7]   site: winding={np.round(np.median(w), 2)} (overlap {_LAST_OVERLAP:.2f}) normals={ns} parity={inside_par:.2f} air_exact={air:.2f}", flush=True)
    else:
        inside = inside_par
        air = float((hdist(S) > 0.5 * pitch).mean())
    return inside, air, site_kind(inside, air) != "INVALID"

def site_kind(inside, air):
    """Two consistent ways a handle can be right, two ways it can be wrong (2026-09-28):
      MEMBRANE: our material where the hull proves AIR   -> add_handle drills it (hard evidence: carving)
      CONTACT : our air gap where the hull is SOLID       -> add_handle joins two touching sheets (LESSONS 25b;
                softer evidence: a visual hull also fills concavities)
      INVALID : material where the hull is solid (drilling real material) or air where the hull is air
                (bridging real empty space) -> never accept."""
    if inside >= 0.6 and air >= 0.6: return "MEMBRANE"
    if inside <= 0.4 and air <= 0.4: return "CONTACT"
    return "INVALID"

CONTACT_GEO_RATIO = float(os.environ.get("CONTACT_GEO_RATIO", "50"))
def contact_geo_ratio(V, F, fi, fj):
    """Surface (geodesic, Dijkstra on the edge graph) distance / straight distance between two faces.
    A true CONTACT joins two DIFFERENT parts pressed together (arm on body): far apart along the
    surface (fertility: ratio 108-560). A crease/crack is the same sheet folded: near along the surface
    too (ratio 1-12). Joining a crease with add_handle adds a spurious tiny handle. Calibrated 2026-09-28
    on 112 GT-labelled candidates (fy1-15): no sample between 11.6 and 108 -> threshold 50."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import dijkstra
    F = np.asarray(F, np.int64); V = np.asarray(V, float)
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); w = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1)
    G = coo_matrix((np.r_[w, w], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(len(V), len(V))).tocsr()
    d = dijkstra(G, indices=[int(x) for x in F[int(fi)]], min_only=True)
    geo = float(d[F[int(fj)]].min()); eu = float(np.linalg.norm(V[F[int(fi)]].mean(0) - V[F[int(fj)]].mean(0)))
    return geo / max(eu, 1e-9)
_LAST_OVERLAP = 0.0; _LAST_WMED = 0.0
def site_kind_full(V, F, fi, fj, inside, air):
    k = site_kind(inside, air)
    if SITE_V2 and _LAST_OVERLAP >= 0.6: k = "CONTACT"      # two parts interpenetrate (winding ~2): join them
    if k == "CONTACT" and contact_geo_ratio(V, F, fi, fj) < CONTACT_GEO_RATIO: return "CREASE"
    return k

R_REJ = float(os.environ.get("R_REJ", "0.3"))   # 2026-09-28: a candidate this close to a REJECTED handle position is skipped
def _rejected_mids():
    return [np.asarray(h["rej_mid"], float) for h in globals().get("prev_handles", [])
            if isinstance(h, dict) and h.get("rej_mid") is not None]
def near_rejected(ci, cj):
    m = 0.5 * (np.asarray(ci, float) + np.asarray(cj, float))
    return any(np.linalg.norm(m - r) < R_REJ for r in _rejected_mids())
def search_handles():
    """prev_handles + the rejected positions as pseudo-handles, so every detector's existing mid-distance
    dedup also steps AWAY from places we already tried and rejected (instead of re-proposing them)."""
    return list(globals().get("prev_handles", [])) + [{"mid": r.tolist(), "blob": None} for r in _rejected_mids()]

def report(tag, V, F):
    ho = heldout_exam(ctx, V, F); wt, _ = check_watertight(F)
    print(f"[{tag}] V={len(V)} F={len(F)} watertight={wt} genus={genus(V, F)} | ho16={ho[0]:.4f} hair={ho[1]} maxblob={ho[2]}", flush=True)

def _snap(title, V, F, hold=30):
    if os.environ.get("SNAPSHOT_DIR"):
        import viz_snap; viz_snap.snap(ctx, mvps, V, F, title, hold=hold)

def find_tunnel_pairs(V, F):
    tri = V[F]; cen = tri.mean(1)
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    E = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); me = np.linalg.norm(V[E[:, 0]] - V[E[:, 1]], axis=1).mean()
    d = hdist(cen)
    cand = np.where(d > OUT_VOX * pitch)[0]
    print(f"[p7] faces outside hull by >{OUT_VOX} voxels: {len(cand)} / {len(F)} (pitch {pitch:.4f}, mean edge {me:.4f})", flush=True)
    import open3d as o3d
    _scene = o3d.t.geometry.RaycastingScene()
    _scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.int32))))
    pairs = []
    for a in range(len(cand)):
        i = cand[a]
        for j in cand[a + 1:]:
            if len(set(F[i]) & set(F[j])): continue                   # must not share vertices
            v = cen[j] - cen[i]; L = np.linalg.norm(v)
            if L > MAX_SEP * me or L < MIN_SEP * me: continue
            if n[i] @ n[j] > FACE_COS: continue                          # facing each other
            # membrane = thin slab of OUR volume inside the GT tunnel: the two faces are
            # back-to-back, outward normals point AWAY from each other
            if n[i] @ v >= 0 or n[j] @ (-v) >= 0: continue
            seg = cen[i] + np.linspace(0.05, 0.95, max(9, int(L / (0.5 * pitch))))[:, None] * v   # sample every half voxel
            if (hdist(seg) < 0.5 * pitch).any(): continue               # whole segment outside hull
            # the segment must not cross OUR mesh either: if it does, it runs through an
            # existing tube/hole (fertility: 5th handle in an already-opened hole -> genus 5 vs GT 4)
            if _scene is not None:
                r = o3d.core.Tensor(np.concatenate([(cen[i] + 0.02 * L * v / L)[None], (v / L)[None]], 1).astype(np.float32))
                hit = _scene.cast_rays(r)["t_hit"].numpy()[0]
                if np.isfinite(hit) and hit < L * 0.98: continue
            # bay vs tunnel: a real through-hole keeps going (hull-free) beyond both faces;
            # an unfilled bay has hull material right behind its bottom face (fertility: 7 handles vs GT 4)
            ext = np.linspace(0.0, EXT_VOX * pitch, 9)[1:]
            back = cen[i] - ext[:, None] * (v / L); fwd = cen[j] + ext[:, None] * (v / L)
            if (hdist(back) < 0.5 * pitch).any() or (hdist(fwd) < 0.5 * pitch).any(): continue
            pairs.append((min(d[i], d[j]) / pitch, -L / me, int(i), int(j)))
    pairs.sort(reverse=True)
    return pairs, cen, d


def project_to_hull(V, iters=3):
    """Hard space-carving step: a vertex outside the voting hull is provably misplaced;
    move it along -grad(dist) onto the hull boundary (trilinear field, few iterations)."""
    P = torch.tensor(V, dtype=torch.float32, device=DEVICE); moved = 0
    for it in range(iters):
        P = P.detach().requires_grad_(True)
        dd = HF.dist(P); m = dd > 0.5 * pitch
        if not m.any(): break
        g, = torch.autograd.grad(dd.sum(), P); g = g / (g.norm(dim=1, keepdim=True) + 1e-9)
        P = P.detach(); P[m] = P[m] - dd[m, None] * g[m]; moved = max(moved, int(m.sum()))
    return P.detach().cpu().numpy().astype(np.float64), moved


def radial_project(V, a0, u, L, tube_verts):
    """Vertices in the tunnel near its medial axis have an ill-defined EDT gradient (it flips
    from vertex to vertex -> faces cross the tunnel and cap it). Instead march each outside
    vertex RADIALLY away from the tunnel axis (a0 + t*u) until it reaches the hull boundary."""
    V = V.copy(); rel = V - a0; t = rel @ u
    radial = rel - t[:, None] * u; r = np.linalg.norm(radial, axis=1)
    d = hdist(V)
    is_tube = np.isin(np.arange(len(V)), list(tube_verts))
    # membrane remnants left around the mouth re-trigger the tunnel test (fertility: 7 handles vs 4);
    # eat them into the mouth by projecting everything outside the hull near the axis radially too
    near = (d > 0.5 * pitch) & (is_tube | ((r < PROJ_RADIUS) & (t > -0.2 * L) & (t < 1.2 * L)))
    idx = np.where(near)[0]
    if len(idx) == 0: return V, 0
    perp1 = np.cross(u, [1.0, 0, 0]); perp1 = perp1 if np.linalg.norm(perp1) > 0.1 else np.cross(u, [0, 1.0, 0]); perp1 /= np.linalg.norm(perp1)
    perp2 = np.cross(u, perp1)
    dirs = radial[idx] / (r[idx, None] + 1e-12)
    deg = r[idx] < 0.5 * pitch
    ang = 2 * np.pi * (idx[deg] % 7) / 7.0
    dirs[deg] = np.cos(ang)[:, None] * perp1 + np.sin(ang)[:, None] * perp2
    steps = np.arange(0.0, 0.5, 0.5 * pitch)
    P = V[idx][:, None, :] + steps[None, :, None] * dirs[:, None, :]
    dd = hdist(P.reshape(-1, 3)).reshape(len(idx), len(steps))
    inside = dd <= 0.5 * pitch
    first = np.where(inside.any(1), inside.argmax(1), 0)   # no hull boundary found along the ray (e.g. through the mouth) -> leave the vertex
    V[idx] = V[idx] + steps[first][:, None] * dirs
    return V, len(idx)



def membrane_patches(V, F, out_vox_override=None):
    """Connected components (edge adjacency) of faces lying outside the voting hull, with the
    Euler characteristic of each patch. A membrane that blocks a tunnel is a topological DISK
    (chi = 1). Once a handle pierces it, the patch becomes an annulus (chi = 0): tunnel already open."""
    from collections import defaultdict
    ov = out_vox_override if out_vox_override is not None else OUT_VOX
    tri = V[F]; cen = tri.mean(1); d = hdist(cen)
    out = np.where(d > ov * pitch)[0]
    if len(out) == 0: return {}, {}
    em = defaultdict(list)
    for fi in out:
        a, b, c = F[fi]
        for e in ((a, b), (b, c), (c, a)): em[(min(e), max(e))].append(fi)
    parent = {int(f): int(f) for f in out}
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for fs in em.values():
        for f2 in fs[1:]: parent[find(int(fs[0]))] = find(int(f2))
    comp = {int(f): find(int(f)) for f in out}
    chi = {}
    for root in set(comp.values()):
        fs = [f for f, r in comp.items() if r == root]
        verts = set(); edges = set()
        for f in fs:
            a, b, c = map(int, F[f]); verts |= {a, b, c}
            for e in ((a, b), (b, c), (c, a)): edges.add((min(e), max(e)))
        chi[root] = len(verts) - len(edges) + len(fs)
    return comp, chi


def absorb_membrane(mesh, ring_ids, membrane_ids, max_iter=100000):
    """Cut the membrane from the mouth outward: repeatedly collapse an edge (ring vertex, membrane
    vertex) with DLFL collapse_edge_tri, survivor placed at the membrane vertex -> the mouth ring
    advances one vertex per collapse until it reaches the membrane rim (which lies on the hull =
    tunnel wall). Euler characteristic preserved (genus fixed by the handle), manifold preserved."""
    from topmod.high_level_ops import collapse_edge_tri
    vlist = list(mesh.vertices.values())
    ring = set(id(vlist[i]) for i in ring_ids)
    memb = set(id(vlist[i]) for i in membrane_ids) - ring
    n = 0; stuck = set(); progress = True
    while progress and n < max_iter:
        progress = False
        for v in list(mesh.vertices.values()):
            if id(v) not in ring or v.id not in mesh.vertices: continue
            for h in list(v.outgoing_halfedges()):
                u = h.twin.origin if h.twin else None
                if u is None or id(u) not in memb or h.edge is None or h.edge.id in stuck: continue
                ux, uy, uz = u.x, u.y, u.z
                sv = collapse_edge_tri(mesh, h.edge)
                if sv is None: stuck.add(h.edge.id); continue
                sv.x, sv.y, sv.z = ux, uy, uz
                memb.discard(id(u)); memb.discard(id(sv)); ring.add(id(sv))
                n += 1; progress = True
                break
    return n, len(memb)


def _is_boundary(v, faces):
    for h in v.outgoing_halfedges():
        if h.face is None or id(h.face) not in faces: return True
    return False

def membrane_to_polygon(mesh, flist, patch_face_idx):
    """Turn a membrane patch (disk of triangles) into ONE polygon face whose boundary is the
    membrane rim: (1) DLFL-collapse every interior vertex into a rim neighbour (survivor placed
    at the rim vertex), (2) delete_edge on every edge shared by two patch faces. Manifold and
    Euler characteristic preserved throughout."""
    from topmod.high_level_ops import collapse_edge_tri
    from topmod.operators import delete_edge
    faces = {id(flist[f]): flist[f] for f in patch_face_idx}
    for _ in range(50):
        interior = [v for v in {id(v): v for f in faces.values() for v in f.vertices()}.values()
                    if v.id in mesh.vertices and not _is_boundary(v, faces)]
        if not interior: break
        moved = False
        for v in interior:
            if v.id not in mesh.vertices: continue
            for pref in (True, False):
                ok = False
                for h in list(v.outgoing_halfedges()):
                    u = h.twin.origin if h.twin else None
                    if u is None or h.edge is None: continue
                    if pref and not _is_boundary(u, faces): continue
                    ux, uy, uz = u.x, u.y, u.z
                    fa, fb = h.face, h.twin.face
                    sv = collapse_edge_tri(mesh, h.edge)
                    if sv is None: continue
                    sv.x, sv.y, sv.z = ux, uy, uz; ok = True; moved = True
                    for f in (fa, fb):
                        if f is not None and f.id not in mesh.faces: faces.pop(id(f), None)
                    break
                if ok: break
        if not moved: break
    faces = {k: f for k, f in faces.items() if f.id in mesh.faces}
    progress = True
    while progress and len(faces) > 1:
        progress = False
        for e in list(mesh.edges.values()):
            if e.id not in mesh.edges: continue
            fa, fb = e.he0.face, e.he1.face
            if fa is None or fb is None or fa is fb: continue
            if id(fa) in faces and id(fb) in faces:
                nf = delete_edge(mesh, e)
                faces.pop(id(fa), None); faces.pop(id(fb), None); faces[id(nf)] = nf
                progress = True
    assert len(faces) == 1, f"membrane did not merge into one polygon ({len(faces)} left)"
    return next(iter(faces.values()))

def open_tunnel_merge(mesh, flist, patch_i, patch_j):
    """membrane_i -> rim polygon, membrane_j -> rim polygon, equalize vertex counts by DLFL
    subdivide_edge on the smaller rim, align the start vertices, add_handle(rim_i, rim_j)."""
    from topmod.high_level_ops import subdivide_edge as dlfl_subdivide_edge
    f1 = membrane_to_polygon(mesh, flist, patch_i)
    f2 = membrane_to_polygon(mesh, flist, patch_j)
    def nverts(f): return len(list(f.halfedges()))
    while nverts(f1) != nverts(f2):
        small = f1 if nverts(f1) < nverts(f2) else f2
        hes = list(small.halfedges())
        longest = max(hes, key=lambda h: (h.origin.x - h.twin.origin.x)**2 + (h.origin.y - h.twin.origin.y)**2 + (h.origin.z - h.twin.origin.z)**2)
        dlfl_subdivide_edge(mesh, longest.edge)
    n = nverts(f1)
    # align: pair verts1[0] with the rim-2 vertex nearest to it (add_handle pairs verts1[t] with reversed verts2[t])
    hes1 = list(f1.halfedges()); v0 = hes1[0].origin
    hes2 = list(f2.halfedges())
    k = min(range(n), key=lambda t: (hes2[t].origin.x - v0.x)**2 + (hes2[t].origin.y - v0.y)**2 + (hes2[t].origin.z - v0.z)**2)
    f2.he = hes2[(k + 1) % n]          # reversed list index n-1 -> hes2[k].origin pairs with v0
    add_handle(mesh, f1, f2)
    return n

def find_tunnel_by_rays(V, F, min_px=int(os.environ.get("MIN_PX", "30")), prev_handles=()):
    """Image-domain space-carving evidence. In a TRAINING view, a background pixel enclosed by
    foreground (a 2D hole in the GT silhouette) proves free space along its whole ray. If our
    mesh is hit by that ray, the entry and exit faces are the two sides of the membrane that
    blocks the tunnel -> add_handle(entry, exit). Holes are ranked by pixel area; the ray is
    taken at the hole's centroid (plus a few fallbacks inside the blob)."""
    import open3d as o3d
    from scipy import ndimage
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.int32))))
    gtn = np.asarray(gt); H, W = gtn.shape[1:]
    cands = []
    for k in range(len(gtn)):
        fg = gtn[k] < 128
        lab, n = ndimage.label(~fg)
        border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
        for c in range(1, n + 1):
            if c in border: continue
            m = lab == c; area = int(m.sum())
            if area < min_px: continue
            rr, cc = np.where(m); order = np.argsort((rr - rr.mean())**2 + (cc - cc.mean())**2)
            cands.append((area, k, rr[order[:5]], cc[order[:5]]))
    cands.sort(key=lambda x: -x[0])
    comp, chi = membrane_patches(V, F)
    print(f"[p7] outside-hull membrane patches: {len(chi)} (disks: {sum(1 for c in chi.values() if c == 1)})", flush=True)
    print(f"[p7] see-through blobs >= {min_px}px across training views: {len(cands)}", flush=True)
    inv = [np.linalg.inv(np.asarray(torch.as_tensor(m).cpu().numpy(), np.float64)) for m in mvps]
    for area, k, rr, cc in cands:
        key = [int(k), int(rr[0]), int(cc[0])]
        if any(h.get("blob") == key for h in prev_handles if isinstance(h, dict)):
            continue   # this see-through blob already got its handle (cascade guard: flaps / narrow tubes)
        for r_, c_ in zip(rr, cc):
            x = (c_ + 0.5) / W * 2 - 1; y = (r_ + 0.5) / H * 2 - 1
            p0 = inv[k] @ np.array([x, y, -1.0, 1.0]); p1 = inv[k] @ np.array([x, y, 1.0, 1.0])
            p0 = p0[:3] / p0[3]; p1 = p1[:3] / p1[3]; dvec = p1 - p0; dvec /= np.linalg.norm(dvec)
            ray = o3d.core.Tensor(np.concatenate([p0, dvec])[None].astype(np.float32))
            h1 = sc.cast_rays(ray); t1 = float(h1["t_hit"].numpy()[0])
            if not np.isfinite(t1): continue
            fi = int(h1["primitive_ids"].numpy()[0])
            ray2 = o3d.core.Tensor(np.concatenate([p1, -dvec])[None].astype(np.float32))
            h2 = sc.cast_rays(ray2); t2 = float(h2["t_hit"].numpy()[0])
            if not np.isfinite(t2): continue
            fj = int(h2["primitive_ids"].numpy()[0])
            if fi == fj or (set(F[fi]) & set(F[fj])): continue
            ci = V[F[fi]].mean(0); cj = V[F[fj]].mean(0)
            if near_rejected(ci, cj): continue   # already tried here and rejected -> look elsewhere
            # topological one-handle-per-tunnel rule: both membrane patches must still be disks
            ki, kj = comp.get(fi), comp.get(fj)
            if ki is None or kj is None or chi[ki] != 1 or chi[kj] != 1:
                print(f"[p7]   skip view {k} hole {area}px: membrane patches chi={None if ki is None else chi[ki]},{None if kj is None else chi[kj]} (not disks -> tunnel already pierced / not a membrane)", flush=True)
                continue
            print(f"[p7] ray evidence: view {k}, hole {area}px, entry face {fi} exit face {fj}, sep {np.linalg.norm(cj-ci):.3f}", flush=True)
            return fi, fj, ci, cj, key
    return None

def find_bridge(V, F, min_vox=int(os.environ.get("BRIDGE_MIN_VOX", "60")), max_try=6):
    """Dual of find_tunnel_by_rays (LESSONS 25): hull-SOLID voxels that the mesh leaves EMPTY, slab-shaped
    (a wall between two tunnels the mesh merged into one opening). add_handle between the two mesh faces at the
    ends of the slab's long axis inserts a bar through the opening; the DR loop inflates it into the wall.
    Returns (fi, fj, ci, cj, key) or None."""
    import open3d as o3d
    from scipy import ndimage as ndi
    S3 = ndi.generate_binary_structure(3, 1)
    Hc = np.asarray(HF.hull).astype(bool)
    Hc = ndi.binary_opening(ndi.binary_closing(Hc, S3, iterations=2), S3, iterations=2)
    lab, n = ndi.label(Hc); Hc = ndi.binary_fill_holes(lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1))
    N = Hc.shape[0]; lo, hi = np.asarray(HF.lo), np.asarray(HF.hi)
    sc = o3d.t.geometry.RaycastingScene()
    sc.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.int32))))
    ax = [np.linspace(lo[i], hi[i], N, dtype=np.float32) for i in range(3)]; M = np.zeros((N, N, N), bool)
    yy, zz = np.meshgrid(ax[1], ax[2], indexing="ij")
    for xi in range(N):
        P = np.stack([np.full_like(yy, ax[0][xi]), yy, zz], -1).reshape(-1, 3)
        M[xi] = (sc.compute_occupancy(o3d.core.Tensor(P)).numpy() > 0.5).reshape(N, N)
    gap = Hc & ~M                                                   # hull material the mesh does not cover
    gl, gn = ndi.label(gap); sz = np.bincount(gl.ravel())[1:]; order = [int(c) for c in np.argsort(-sz) if sz[c] >= min_vox][:40]
    world = lambda v: lo + np.asarray(v, float) / (N - 1) * (hi - lo)
    def free_both_sides(cen_v, nrm, t):
        """A WALL between two tunnels has hull-FREE space on both broad sides within a few voxels; hull slack
        (a concavity the hull cannot carve) has free space on one side only."""
        ok = []
        for sgn in (+1.0, -1.0):
            hit = False
            for d in np.arange(t / 2 + 1, t / 2 + 12, 1.0):
                q = np.round(cen_v + sgn * d * nrm).astype(int)
                if (q < 0).any() or (q >= N).any(): hit = True; break
                if M[q[0], q[1], q[2]]: break                       # ran into mesh material first -> not free on this side
                if not Hc[q[0], q[1], q[2]]: hit = True; break     # outside the hull -> free space
            ok.append(hit)
        return all(ok)
    walls = []
    for c in order:
        pts = np.argwhere(gl == c + 1).astype(float); cen_v = pts.mean(0); u, s, vt = np.linalg.svd(pts - cen_v, full_matrices=False)
        ext = 2 * s / np.sqrt(len(pts))
        if ext[2] > 12 or ext[0] < 6: continue                     # a wall is thin (<12 vox) and has some extent
        if free_both_sides(cen_v, vt[2], ext[2]): walls.append((c, pts, cen_v, vt, ext))
    print(f"[p7] bridge evidence: hull-solid/mesh-empty components >= {min_vox} vox: {len(order)}; thin + free on both sides (walls): {len(walls)} (sizes {[int(sz[w[0]]) for w in walls]})", flush=True)
    for c, pts, cen_v, vt, ext in walls[:max_try]:
        c0 = world(cen_v); a = vt[0] * np.sign(vt[0] @ (hi - lo))   # long axis of the slab, world frame (grid is axis-aligned)
        a = a / (np.linalg.norm(a) + 1e-12); L = float(ext[0]) * float(HF.pitch)
        key = ["bridge", int(c + 1), [round(float(x), 3) for x in c0]]
        if any(h.get("blob") == key for h in prev_handles if isinstance(h, dict)): continue
        hits = []
        for sgn in (+1.0, -1.0):
            ray = o3d.core.Tensor(np.concatenate([c0, sgn * a])[None].astype(np.float32)); h = sc.cast_rays(ray)
            t = float(h["t_hit"].numpy()[0]); hits.append((t, int(h["primitive_ids"].numpy()[0])) if np.isfinite(t) else None)
        if any(h is None for h in hits): print(f"[p7]   slab {c+1} ({int(sz[c])} vox, extent {np.round(ext,1)} vox): long-axis ray misses the mesh on one side -> skip", flush=True); continue
        (ti, fi), (tj, fj) = hits
        if ti + tj > 3.0 * L + 6 * HF.pitch: print(f"[p7]   slab {c+1}: end faces too far apart ({(ti+tj)/HF.pitch:.0f} vox vs slab {ext[0]:.0f}) -> skip", flush=True); continue
        if fi == fj or (set(F[fi]) & set(F[fj])): continue
        ci, cj = V[F[fi]].mean(0), V[F[fj]].mean(0)
        print(f"[p7] bridge evidence: slab {c+1} {int(sz[c])} vox at {np.round(c0,3)}, extent {np.round(ext,1)} vox, end faces {fi},{fj} (gaps {ti/HF.pitch:.0f}/{tj/HF.pitch:.0f} vox)", flush=True)
        return fi, fj, ci, cj, key
    return None

def find_contact_join(V, F, prev_handles=(), r_vox=float(os.environ.get("CONTACT_R_VOX", "1.5")), cos_max=-0.7):
    """LESSONS 25b: a handle that no image can see = two surface sheets pressed together but not joined
    (fertility 4th tunnel wall). Find non-adjacent face pairs with opposed normals within r_vox voxels,
    cluster them, and return the best pair of the largest cluster for a zero-length add_handle (= join)."""
    from scipy.spatial import cKDTree
    tri = V[F]; cen = tri.mean(1); nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    vf = {}
    for k, f in enumerate(F):
        for x in f: vf.setdefault(int(x), set()).add(k)
    ring = [set().union(*[vf[int(x)] for x in f]) for f in F]                       # faces sharing a vertex (1-ring)
    ringv = lambda k: set(int(x) for q in ring[k] for x in F[q])                        # vertices of the 1-ring
    disjoint = lambda a, b: not (ringv(a) & set(map(int, F[b]))) and not (ringv(b) & set(map(int, F[a])))
    pairs = cKDTree(cen).query_pairs(r=r_vox * pitch, output_type="ndarray")
    ok = [(int(i), int(j)) for i, j in pairs if j not in ring[i] and nrm[i] @ nrm[j] < cos_max
          and abs((cen[j] - cen[i]) @ nrm[i]) > 0.3 * np.linalg.norm(cen[j] - cen[i])]
    print(f"[p7] contact search: {len(pairs)} face pairs within {r_vox} vox, {len(ok)} opposed-normal non-adjacent contacts", flush=True)
    if not ok: return None
    P = np.array([(cen[i] + cen[j]) / 2 for i, j in ok]); parent = list(range(len(P)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in cKDTree(P).query_pairs(r=4 * pitch, output_type="ndarray"): parent[find(int(a))] = find(int(b))
    roots = np.array([find(i) for i in range(len(P))])
    for r in sorted(set(roots.tolist()), key=lambda r: -(roots == r).sum()):
        m = np.where(roots == r)[0]; mid = P[m].mean(0); key = ["contact", [round(float(x), 2) for x in mid]]
        if any(h.get("blob") == key for h in prev_handles if isinstance(h, dict)): continue
        if any(np.linalg.norm(mid - r) < R_REJ for r in _rejected_mids()):
            print(f"[p7]   contact cluster at {np.round(mid, 3)}: near a rejected position -> skip", flush=True); continue
        best = sorted(m, key=lambda q: nrm[ok[q][0]] @ nrm[ok[q][1]])                   # most opposed first
        for q in best:
            i, j = ok[q]
            if disjoint(i, j):
                print(f"[p7] contact evidence: cluster of {len(m)} pairs at {np.round(mid, 3)}, join faces {i},{j} (dist {np.linalg.norm(cen[j]-cen[i])/pitch:.2f} vox, n.n {nrm[i]@nrm[j]:.2f})", flush=True)
                return i, j, cen[i], cen[j], key
        print(f"[p7]   contact cluster at {np.round(mid, 3)}: no pair with disjoint 1-rings -> skip", flush=True)
    return None


def find_near_pairs(V, F, prev_handles=(), r_vox=float(os.environ.get("THIN_R_VOX", "6")), cos_max=-0.7, all_clusters=False):
    """2026-09-30 (Boss): two faces CLOSE in space but FAR along the surface (non-adjacent), with opposed normals, are
    the two sides of a thin sheet (back-to-back: a crushed membrane) or of a contact (face-to-face / interpenetrating).
    On the refined mesh this finds in seconds what the hull face-pair search needs ~15 min for. Clusters of such
    pairs are classified by the site check; only MEMBRANE / CONTACT clusters are returned (a real thin part - a
    plate, an ear - has no hull air through it and is dropped)."""
    from scipy.spatial import cKDTree
    tri = V[F]; cen = tri.mean(1); nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    vf = {}
    for k, f in enumerate(F):
        for x in f: vf.setdefault(int(x), set()).add(k)
    ring = lambda k: set().union(*[vf[int(x)] for x in F[k]])
    ringv = lambda k: set(int(x) for q in ring(k) for x in F[q])
    disjoint = lambda a, b: not (ringv(a) & set(map(int, F[b]))) and not (ringv(b) & set(map(int, F[a])))
    pairs = cKDTree(cen).query_pairs(r=r_vox * pitch, output_type="ndarray")
    if len(pairs):
        i, j = pairs[:, 0], pairs[:, 1]; dv = cen[j] - cen[i]; dn = np.linalg.norm(dv, axis=1) + 1e-12
        m = (np.einsum("ij,ij->i", nrm[i], nrm[j]) < cos_max) & (np.abs(np.einsum("ij,ij->i", dv, nrm[i])) > float(os.environ.get("THIN_ALIGN", "0.7")) * dn)
        pairs = pairs[m]
    ok = [(int(i), int(j)) for i, j in pairs if j not in ring(int(i))]
    print(f"[p7] near-pair search: {len(ok)} opposed-normal non-adjacent face pairs within {r_vox:g} vox", flush=True)
    if not ok: return []
    P = np.array([(cen[i] + cen[j]) / 2 for i, j in ok]); parent = list(range(len(P)))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for a, b in cKDTree(P).query_pairs(r=4 * pitch, output_type="ndarray"): parent[find(int(a))] = find(int(b))
    roots = np.array([find(q) for q in range(len(P))]); out = []
    for r in sorted(set(roots.tolist()), key=lambda r: -(roots == r).sum()):
        m = np.where(roots == r)[0]; mid = P[m].mean(0); key = ["near", [round(float(x), 2) for x in mid]]
        if len(m) < int(os.environ.get("THIN_MIN_PAIRS", "3")): continue
        if not all_clusters:
            if any(isinstance(h, dict) and (h.get("blob") == key or (h.get("mid") is not None and np.linalg.norm(np.asarray(h["mid"], float) - mid) < R_DEDUP)) for h in prev_handles): continue
            if any(np.linalg.norm(mid - q) < R_REJ for q in _rejected_mids()): continue
        best = sorted(m, key=lambda q: np.linalg.norm(P[q] - mid))[:40]                  # pairs nearest the cluster centre
        hit = next(((ok[q][0], ok[q][1]) for q in best if disjoint(*ok[q])), None)
        if hit is None: continue
        i, j = hit; _in, _air, _ = membrane_check(V, F, cen[i], cen[j], fi=i, fj=j); kind = site_kind_full(V, F, i, j, _in, _air)
        print(f"[p7]   near cluster of {len(m)} pairs at {np.round(mid, 3)}: faces {i},{j} dist {np.linalg.norm(cen[j]-cen[i])/pitch:.2f} vox inside={_in:.2f} air={_air:.2f} -> {kind}", flush=True)
        if kind in ("MEMBRANE", "CONTACT") and not all_clusters and int(os.environ.get("RANK_PREFILTER", "0")):
            # strict topology: the loop this handle would close (segment + surface path) must enclose a hull tunnel
            # that the surface does not realise yet (linking vector independent of the existing handle loops)
            try:
                import linking_audit as _la
                if "_np_air" not in globals():
                    globals()["_np_air"] = _la.air_loops(SHAPE, log=lambda m: None)
                _air_l = globals()["_np_air"]
                if _air_l:
                    _R = _la.surface_matrix(V, F, _air_l)[2]; _v = _la.candidate_vector(V, F, i, j, _air_l)
                    _r0 = int(np.linalg.matrix_rank(_R)) if _R.size else 0; _r1 = int(np.linalg.matrix_rank(np.vstack([_R, _v[None]])))
                    print(f"[p7]   linking vector {_v.tolist()}: tunnels realised {_r0} -> {_r1} {'(new tunnel)' if _r1 > _r0 else '(no new tunnel -> skip)'}", flush=True)
                    if _r1 <= _r0: continue
            except Exception as _e: print(f"[p7]   linking prefilter failed ({_e}); candidate kept", flush=True)
        if kind in ("MEMBRANE", "CONTACT") or all_clusters: out.append((i, j, cen[i], cen[j], key, kind))
    if not all_clusters: out.sort(key=lambda c: c[5] != "MEMBRANE")     # stable: membranes (hard hull-air evidence) before contacts
    return out


def find_tunnel_by_hull(V, F, HF_in, prev_handles=(), g_target=None):
    """Hull-located tunnel plugs -> add_handle face pairs. Implementation: hull_locate.py
    (closing-radius ladder with frozen claimed plugs, skeleton centreline, occupancy-walk face pairs)."""
    import hull_locate
    return hull_locate.find_tunnel_by_hull(V, F, HF_in, prev_handles, g_target, res=HULL_LOC_RES,
                                           cache=HULL_PLUGS_CACHE, genus_fn=genus, r_dedup=R_DEDUP,
                                           viz_dir=os.environ.get("HULL_VIZ_DIR"),
                                           log=lambda m: print(m, flush=True))


report("base", V, Fa); _snap(f"Stage 7 [genus discovery] base genus={genus(V, Fa)}", V, Fa)
import json
prev_handles = json.load(open(HANDLES_JSON)) if HANDLES_JSON and os.path.exists(HANDLES_JSON) else []
import handle_guard as _hg
_HAXES = []                                             # accepted tunnel axes (a0, u, L) for the anti-collapse guard
_bx = BASE_NPZ + ".haxes.npz"
if os.path.exists(_bx):
    _z = np.load(_bx); _HAXES = [(_z[f"a0_{k}"], _z[f"u_{k}"], float(_z[f"L_{k}"])) for k in range(int(_z["n"]))]
n_added = 0
if int(os.environ.get("THIN_DRY", "0")):      # debug: enumerate every near-pair cluster with its site kind, then exit
    import time as _t; _t0 = _t.time(); _c = find_near_pairs(V, Fa, all_clusters=True)
    print(f"[p7] THIN_DRY: {len(_c)} cluster(s), {sum(c[5] in ('MEMBRANE', 'CONTACT') for c in _c)} site-valid, {_t.time() - _t0:.1f}s (mesh genus {genus(V, Fa)})", flush=True)
    _rows = []
    for c in _c:
        _in, _air, _ = membrane_check(V, Fa, c[2], c[3], fi=c[0], fj=c[1])
        _rows.append({"ci": np.asarray(c[2], float).tolist(), "cj": np.asarray(c[3], float).tolist(), "kind": c[5], "inside": _in, "air": _air,
                      "w_med": globals().get("_LAST_WMED"), "overlap": globals().get("_LAST_OVERLAP"), "geo_ratio": float(contact_geo_ratio(V, Fa, c[0], c[1])), "mesh_genus": int(genus(V, Fa))})
    json.dump(_rows, open(f"/tmp/thin_dry_{os.path.basename(BASE_NPZ)}.json", "w")); sys.exit(0)
MODE_BRIDGE = False
# When DETECT=hull, override MAX_HANDLES so phase7_multi.sh's MAX_HANDLES=1
# doesn't prevent adding multiple hull handles in one invocation (spec: "add ALL
# returned handles in that round").  find_tunnel_by_hull is called fresh each
# iteration because add_handle modifies V/Fa and invalidates face indices.
_hull_max = MAX_HANDLES   # one handle per round so phase7_multi.sh can verify (DR) and revert each one
# BATCH_OPEN=1 (2026-09-30, Boss): open every site-valid candidate in ONE invocation, re-detecting on the
# updated mesh after each add_handle and with NO DR in between (so waiting membranes are not crushed by DR,
# and Stage 3 needs one DR loop instead of one per handle). A candidate that is not MEMBRANE/CONTACT is
# skipped for this batch only (not persisted: the sequential rounds afterwards may still try it with DR
# verification). Several candidates can belong to one tunnel; re-detection after each handle handles that.
BATCH_OPEN = int(os.environ.get("BATCH_OPEN", "0"))
if BATCH_OPEN: _hull_max = MAX_HANDLES + int(os.environ.get("BATCH_TRIES", "8"))
def _save_handles():
    if HANDLES_JSON: json.dump([h for h in prev_handles if not (isinstance(h, dict) and h.get("batch_skip"))], open(HANDLES_JSON, "w"))
for k in range(_hull_max):
    MODE_BRIDGE = False
    if BATCH_OPEN and n_added >= MAX_HANDLES: break
    if G_TARGET is not None and genus(V, Fa) >= G_TARGET:
        print(f"[p7] genus {genus(V, Fa)} == target g*={G_TARGET}: no more handles", flush=True); break
    if DETECT in ("hull", "membrane"):
        if DETECT == "membrane":
            import membrane_locate
            _hull_cands = membrane_locate.find_tunnel_by_membranes(V, Fa, HF, search_handles(), G_TARGET, r_dedup=R_DEDUP, log=lambda m: print(m, flush=True))
        else:
            _hull_cands = find_tunnel_by_hull(V, Fa, HF, search_handles(), G_TARGET)
        if _hull_cands:
            hit = _hull_cands[0]
            i, j, _ci, _cj, _blob = hit
            tri = V[Fa]; cen = tri.mean(1); d = hdist(cen)
            negL = -np.linalg.norm(_cj - _ci) / np.linalg.norm(V[Fa[:, 0]] - V[Fa[:, 1]], axis=1).mean()
            print(f"[p7] tunnel-evidence pairs: 1 ({DETECT})", flush=True)
        elif G_TARGET is not None and genus(V, Fa) < G_TARGET:
            hit = None; _hc_hit = False
            # HULL-GUIDED COMPLETION: membrane detection stalled (at the refined stage the mesh is flush to the
            # hull, so no face sits OUTSIDE it and there is no outside-hull membrane to find) -- but the oracle
            # says a tunnel is still missing. Ask the HULL itself where it is (closing-ladder plug analysis on the
            # carved hull is independent of mesh refinement). These candidates carry key[0]=="hull", so the
            # count-first gate (Rule C') accepts them unconditionally via rescue-hull.
            _np_hit = False
            if int(os.environ.get("NEAR_PAIRS", "0")):
                _npc = find_near_pairs(V, Fa, search_handles())
                if _npc:
                    hit = _npc[0][:5]; _np_hit = True; MODE_BRIDGE = _npc[0][5] == "CONTACT"   # contact = plain single-face join
                    print(f"[p7] tunnel-evidence pairs: 1 (near-pair, {_npc[0][5]}; {len(_npc)} site-valid cluster(s))", flush=True)
            if hit is None and DETECT == "membrane" and int(os.environ.get("HULL_COMPLETE", "1")):
                # memo: after a rejected round phase7_multi reverts to the SAME mesh, so the (slow, ~15 min)
                # hull face-pair query would be recomputed on identical input. Key = mesh + prior handles.
                import hashlib, pickle
                _mk = hashlib.sha1(np.ascontiguousarray(V, np.float64).tobytes() + np.ascontiguousarray(Fa, np.int64).tobytes()
                                   + json.dumps(prev_handles, sort_keys=True, default=str).encode()).hexdigest()[:16]
                _mf = os.path.join(OUTD, "hc_memo", f"{SHAPE}_{_mk}.pkl")
                if os.path.exists(_mf):
                    _hc = pickle.load(open(_mf, "rb")); print(f"[p7] hull-completion memo hit ({len(_hc)} validated candidate(s))", flush=True)
                else:
                    _hc = find_tunnel_by_hull(V, Fa, HF, search_handles(), G_TARGET)
                    _hc_ok = []
                    for _c in _hc:
                        _in, _air, _okm = membrane_check(V, Fa, _c[2], _c[3], fi=_c[0], fj=_c[1])
                        _k = site_kind_full(V, Fa, _c[0], _c[1], _in, _air)
                        print(f"[p7] hull-completion candidate faces {_c[0]},{_c[1]}: inside={_in:.2f} air={_air:.2f} -> {_k}", flush=True)
                        if _k in ("MEMBRANE", "CONTACT"): _hc_ok.append(_c)
                    _hc = _hc_ok
                    os.makedirs(os.path.dirname(_mf), exist_ok=True); pickle.dump(_hc, open(_mf, "wb"))
                if _hc:
                    hit = _hc[0]; _hc_hit = True
                    print(f"[p7] hull-guided completion: {len(_hc)} plug(s) located on the hull, genus {genus(V, Fa)} < g*={G_TARGET}", flush=True)
                    print(f"[p7] tunnel-evidence pairs: 1 (hull-completion)", flush=True)
            if hit is None:
                # Hull candidates exhausted or no valid face pair: fall back to rays
                print(f"[p7] hull: 0 valid candidates, genus {genus(V, Fa)} < g*={G_TARGET}: rays fallback", flush=True)
                for _lv, (_mp, _ov) in enumerate(RELAX):
                    OUT_VOX = _ov
                    hit = find_tunnel_by_rays(V, Fa, min_px=int(_mp), prev_handles=prev_handles)
                    if hit is not None:
                        if _lv > 0: print(f"[p7] rays fallback at relaxation level {_lv} (min_px {_mp}, out_vox {_ov})", flush=True)
                        break
            if hit is None and int(os.environ.get("BRIDGE", "0")):
                hit = find_bridge(V, Fa)
                if hit is not None: MODE_BRIDGE = True
            if hit is None and int(os.environ.get("CONTACT", "1")):
                hit = find_contact_join(V, Fa, prev_handles=search_handles())
                if hit is not None: MODE_BRIDGE = True
            if hit is None:
                print(f"[p7] tunnel-evidence pairs: 0 (genus {genus(V, Fa)} < g*={G_TARGET}: UNREACHED)", flush=True); break
            i, j, _ci, _cj, _blob = hit
            tri = V[Fa]; cen = tri.mean(1); d = hdist(cen)
            negL = -np.linalg.norm(_cj - _ci) / np.linalg.norm(V[Fa[:, 0]] - V[Fa[:, 1]], axis=1).mean()
            if not _hc_hit and not _np_hit: print(f"[p7] tunnel-evidence pairs: 1 (ray fallback)", flush=True)
        else:
            print(f"[p7] tunnel-evidence pairs: 0", flush=True); break
    elif DETECT == "rays":
        hit = None
        for _lv, (_mp, _ov) in enumerate(RELAX if G_TARGET is not None else RELAX[:1]):
            OUT_VOX = _ov
            hit = find_tunnel_by_rays(V, Fa, min_px=int(_mp), prev_handles=prev_handles)
            if hit is not None:
                if _lv > 0: print(f"[p7] evidence found at relaxation level {_lv} (min_px {_mp}, out_vox {_ov}) because genus {genus(V, Fa)} < g*={G_TARGET}", flush=True)
                break
        if hit is None and G_TARGET is not None and genus(V, Fa) < G_TARGET and int(os.environ.get("BRIDGE", "0")):
            hit = find_bridge(V, Fa)
            if hit is not None: MODE_BRIDGE = True
        if hit is None and G_TARGET is not None and genus(V, Fa) < G_TARGET and int(os.environ.get("CONTACT", "1")):
            hit = find_contact_join(V, Fa, prev_handles=search_handles())
            if hit is not None: MODE_BRIDGE = True                                   # plain single-face add_handle (no membrane merge)
        if hit is None: print(f"[p7] tunnel-evidence pairs: 0" + (f" (genus {genus(V, Fa)} < g*={G_TARGET}: UNREACHED)" if G_TARGET is not None else ""), flush=True); break
        i, j, _ci, _cj, _blob = hit
        tri = V[Fa]; cen = tri.mean(1); d = hdist(cen); negL = -np.linalg.norm(_cj - _ci) / np.linalg.norm(V[Fa[:, 0]] - V[Fa[:, 1]], axis=1).mean()
        print(f"[p7] tunnel-evidence pairs: 1 (ray)", flush=True)
    else:
        pairs, cen, d = find_tunnel_pairs(V, Fa)
        print(f"[p7] tunnel-evidence pairs: {len(pairs)}", flush=True)
        if not pairs: break
        score, negL, i, j = pairs[0]
    if DRY:
        print(f"[p7] DRY: best pair {i},{j} out {d[i]/pitch:.1f}/{d[j]/pitch:.1f} vox, sep {-negL:.2f} edges, centroids {np.round(cen[i],2)} {np.round(cen[j],2)}", flush=True); break
    _mi, _ma, _mok = membrane_check(V, Fa, cen[i], cen[j], fi=i, fj=j)
    _mk = site_kind_full(V, Fa, i, j, _mi, _ma)
    try:
        with open(os.path.join(OUTD, f"sitelog_{SHAPE}.jsonl"), "a") as _sf:
            _sf.write(json.dumps({"base": os.path.basename(BASE_NPZ), "mesh_genus": int(genus(V, Fa)), "fi": int(i), "fj": int(j),
                                  "ci": np.asarray(cen[i], float).tolist(), "cj": np.asarray(cen[j], float).tolist(),
                                  "inside": _mi, "air": _ma, "w_med": globals().get("_LAST_WMED", None), "overlap": globals().get("_LAST_OVERLAP", None),
                                  "geo_ratio": float(contact_geo_ratio(V, Fa, i, j)), "site": _mk,
                                  **({"decision": "batch_accept" if _mk in ("MEMBRANE", "CONTACT") else "batch_skip"} if BATCH_OPEN else {})}) + "\n")
    except Exception as _e: print(f"[p7] sitelog write failed: {_e}", flush=True)
    _gr = f" geo_ratio={contact_geo_ratio(V, Fa, i, j):.0f}" if site_kind(_mi, _ma) == "CONTACT" else ""
    print(f"[p7] membrane check: inside={_mi:.2f} air={_ma:.2f}{_gr} -> {_mk}", flush=True)
    # bt1 (2026-09-30): a LONG tube (sep 7 edges) bored through a thick slab of wrong material reached genus 4 with the
    # base arch still sealed (redundant with a short drill in the same region). Without DR between handles nothing can
    # tell two membranes of one tunnel apart, so batch only opens THIN membranes (sep <= BATCH_SEP_MAX edges); thick
    # ones are left to the DR-verified sequential rounds.
    _batch_thick = BATCH_OPEN and (-negL) > float(os.environ.get("BATCH_SEP_MAX", "3"))
    if BATCH_OPEN and (_mk not in ("MEMBRANE", "CONTACT") or _batch_thick):
        prev_handles.append({"mid": None, "rej_mid": ((cen[i] + cen[j]) / 2).tolist(), "rejected": True, "blob": None, "batch_skip": True})
        print(f"[p7] batch: skip {'THICK (sep %.1f edges)' % (-negL) if _batch_thick else _mk} candidate at {np.round((cen[i] + cen[j]) / 2, 3)} (left for the verified sequential rounds)", flush=True)
        continue
    print(f"[p7] add_handle between faces {i},{j}: out {d[i]/pitch:.1f}/{d[j]/pitch:.1f} vox, sep {-negL:.2f} edges, centroids {np.round(cen[i],3)} {np.round(cen[j],3)}", flush=True)
    # Thin/pinched membrane: entry and exit faces are on different sheets but their 1-rings overlap
    # (sheets touch at 1-ring distance) -> the tube's side quads would duplicate existing edges and break
    # watertightness (fertility 4th tunnel: sep 0.9 edges, ring overlap 10). Step each face away from the
    # other sheet: pick the nearest face (by centroid) on the same side whose vertex 1-ring is disjoint
    # from the other face's vertex 1-ring.
    _vf = {}
    for _k, _f in enumerate(Fa):
        for _x in _f: _vf.setdefault(int(_x), []).append(_k)
    def _ringverts(f):                       # vertices of all faces sharing a vertex with face f
        return set(int(x) for v in Fa[f] for k in _vf[int(v)] for x in Fa[k])
    def _disjoint(a, b): return not (_ringverts(a) & set(int(x) for x in Fa[b])) and not (_ringverts(b) & set(int(x) for x in Fa[a]))
    if not _disjoint(i, j):
        _i0, _j0 = i, j
        _ni = np.argsort(np.linalg.norm(cen - cen[i], axis=1))[:80]; _nj = np.argsort(np.linalg.norm(cen - cen[j], axis=1))[:80]
        _ni = [int(k) for k in _ni if d[k] > 0]; _nj = [int(k) for k in _nj if d[k] > 0]     # stay outside the hull (on the membrane)
        _pairs = [(a_, b_) for a_ in _ni for b_ in _nj if _disjoint(a_, b_)]
        if _pairs:
            i, j = min(_pairs, key=lambda ab: np.linalg.norm(cen[ab[0]] - cen[_i0]) + np.linalg.norm(cen[ab[1]] - cen[_j0]))
        print(f"[p7] pinched membrane (1-rings overlapped): stepped faces {_i0},{_j0} -> {i},{j}; candidates={len(_pairs)}, new sep {np.linalg.norm(cen[j]-cen[i])/np.linalg.norm(V[Fa[:, 0]] - V[Fa[:, 1]], axis=1).mean():.2f} edges", flush=True)
    def _load_mesh():
        with tempfile.NamedTemporaryFile("w", suffix=".obj", delete=False) as fh:
            for x, y, zz in V: fh.write(f"v {x} {y} {zz}\n")
            for a, b, c in Fa: fh.write(f"f {a+1} {b+1} {c+1}\n")
            path = fh.name
        m = from_obj(path); os.unlink(path)
        return m, list(m.iter_faces())
    mesh, faces = _load_mesh()
    # membrane vertex sets (front patch of face i, back patch of face j) BEFORE the handle
    if ABSORB:
        comp_, chi_ = membrane_patches(V, Fa)
        memb_faces = [f for f, r in comp_.items() if r in (comp_.get(i), comp_.get(j))]
        memb_ids = set(int(x) for f in memb_faces for x in Fa[f])
        ring_ids = set(int(x) for x in Fa[i]) | set(int(x) for x in Fa[j])
    if OPEN == "merge" and not MODE_BRIDGE:
        comp_, chi_ = membrane_patches(V, Fa)
        pi = [f for f, r in comp_.items() if r == comp_.get(i)]; pj = [f for f, r in comp_.items() if r == comp_.get(j)]
        if comp_.get(i) is None: pi = [i]
        if comp_.get(j) is None: pj = [j]
        if comp_.get(i) is not None and comp_.get(i) == comp_.get(j):
            print("[p7] entry and exit faces lie in the SAME membrane patch (thin sheet): using single faces", flush=True); pi, pj = [i], [j]
        try:
            nrim = open_tunnel_merge(mesh, faces, pi, pj)
            for f in list(mesh.faces.values()):
                if len(f.vertices()) > 3: dlfl_stellate(mesh, f)
            _vv, _ff = to_triangle_arrays(mesh)
            _wt, _nb = check_watertight(np.asarray(_ff, np.int64))
            if not _wt: raise RuntimeError(f"merge left {_nb} bad edges")
            print(f"[p7] membranes merged to rim polygons ({len(pi)}+{len(pj)} faces) -> handle with {nrim}-gon rims", flush=True)
        except Exception as ex:
            # thin sheets: front/back rims can share vertices -> non-manifold. Fall back to the plain
            # single-face handle on a fresh mesh (never leave a broken mesh behind).
            print(f"[p7] merge failed ({ex}); falling back to single-face add_handle", flush=True)
            mesh, faces = _load_mesh()
            add_handle(mesh, faces[i], faces[j])
    else:
        add_handle(mesh, faces[i], faces[j])
    if ABSORB:
        n_abs, left = absorb_membrane(mesh, ring_ids, memb_ids)
        print(f"[p7] membrane absorbed into the mouth: {n_abs} DLFL collapses ({len(memb_ids)} membrane verts, {left} left)", flush=True)
    for f in list(mesh.faces.values()):
        if len(f.vertices()) > 3: dlfl_stellate(mesh, f)
    vv, ff = to_triangle_arrays(mesh)
    V2, F2 = np.asarray(vv, float), np.asarray(ff, np.int64)
    wt, nbad = check_watertight(F2)
    if not wt:
        print(f"[p7] handle broke watertightness ({nbad} bad edges): blacklisting this blob and continuing", flush=True)
        prev_handles.append({"mid": None, "blob": _blob if DETECT == "rays" else None})
        _save_handles()
        continue
    n_before = len(V) if not (ABSORB or OPEN == 'merge') else -1
    if ABSORB or OPEN == 'merge':
        # after collapses vertex order changed: tube verts = those outside the hull among the new mesh
        d2 = hdist(V2); tube_verts = set(np.where(d2 > 0.5 * pitch)[0].tolist())
    else:
        tube_verts = set(map(int, Fa[i])) | set(map(int, Fa[j])) | set(range(n_before, len(V2)))
    V, Fa = V2, F2
    a0 = cen[i]; u = cen[j] - cen[i]; L_ = float(np.linalg.norm(u))
    if L_ < 1e-6: u = np.cross(V[Fa[i][1]] - V[Fa[i][0]], V[Fa[i][2]] - V[Fa[i][0]]); u /= np.linalg.norm(u) + 1e-12; L_ = 1e-6   # coincident faces (pinched membrane): axis = face normal
    else: u = u / L_   # tunnel axis from the two face centroids
    if PROJECT:
        # project FIRST (thin tube -> tunnel wall), then refine on the wall, re-project each level.
        # Refining the thin tube before projecting produced sliver fans that crossed when inflated.
        V, mv = radial_project(V, a0, u, L_, tube_verts); print(f"[p7] radial projection (pre-refine): moved {mv} verts", flush=True)
    for _ in range(TUBE_SUBDIV):
        # a 3-edge-wide tube spanning a thick slab has no interior vertices for the hull
        # field to act on (3holes: 10 edges long, hole never opened). Refine the tube.
        from phase1c_pipeline import dlfl_subdivide_arrays
        fids = [k for k, f in enumerate(Fa) if all(int(x) in tube_verts for x in f)]
        nb = len(V)
        V, Fa, ne = dlfl_subdivide_arrays(V, Fa, fids, expand_ring=False)
        tube_verts |= set(range(nb, len(V)))
        wt, nbad = check_watertight(Fa); assert wt, nbad
        print(f"[p7] tube refine: {len(fids)} faces, split {ne} edges -> V={len(V)} F={len(Fa)}", flush=True)
        if PROJECT:
            V, mv = radial_project(V, a0, u, L_, tube_verts); print(f"[p7] radial projection: moved {mv} verts", flush=True)
    n_added += 1
    _HAXES.append((np.asarray(a0,float), np.asarray(u,float)/(np.linalg.norm(u)+1e-12), float(L_)))
    prev_handles.append({"mid": ((cen[i] + cen[j]) / 2).tolist(), "blob": _blob if DETECT in ("rays", "hull", "membrane") else None})
    _save_handles()
    _loops = _hg.tree_cotree_loops(V, Fa); _g = genus(V, Fa)
    print(f"[p7] H1 verification: {len(_loops)} generator loops = 2*genus? (2*{_g}={2*_g})  {'OK' if len(_loops)==2*_g else 'MISMATCH'}", flush=True)
    report(f"after handle {n_added}", V, Fa); _snap(f"Stage 7 [DLFL add_handle #{n_added}] genus={genus(V, Fa)}", V, Fa, hold=45)
np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz", verts=V, tris=Fa)
if _HAXES:
    _hx = {"n": len(_HAXES)}
    for k,(a0,u,L) in enumerate(_HAXES): _hx[f"a0_{k}"]=a0; _hx[f"u_{k}"]=u; _hx[f"L_{k}"]=L
    np.savez_compressed(f"{OUTD}/cow_{SHAPE}_{TAG}.npz.haxes.npz", **_hx)
    print(f"[p7] saved {len(_HAXES)} handle axes for the guard", flush=True)
print(f"[p7] handles added: {n_added}; saved cow_{SHAPE}_{TAG}.npz | genus {genus(V, Fa)}" + (f" / target {G_TARGET}" if G_TARGET is not None else ""), flush=True)
