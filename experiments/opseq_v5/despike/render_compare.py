"""Render an N-column x 4-row visual comparison (full az150 / full az-30 /
hands closeup / back closeup) of meshes given as label=path (npz or obj, all in
our normalized frame). Flat-shaded with nvdiffrast, mesh edges not drawn.
Usage: python3 despike/render_compare.py OUT.png "GT=path" "label=path" ...
"""
import sys, os, numpy as np, torch
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5")
sys.path.insert(0, "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike")
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw, ImageFont
from eval_local_refine import load_obj, normalize_to_range, BUNNY_PATH
DEV = "cuda"
ctx = dr.RasterizeCudaContext()

def load(p):
    if p == "GT":
        v, f = load_obj(os.path.join(os.path.dirname(BUNNY_PATH), "armadillo.obj")); return normalize_to_range(v), np.asarray(f)
    if p.endswith(".npz"):
        z = np.load(p); return z["verts"].astype(float), z["tris"].astype(np.int64)
    v, f = load_obj(p); return np.asarray(v, float), np.asarray(f, np.int64)

def lookat(eye, ctr, up=(0, 1, 0)):
    eye, ctr, up = map(np.asarray, (eye, ctr, up)); f = ctr - eye; f /= np.linalg.norm(f)
    s = np.cross(f, up); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4); M[0, :3], M[1, :3], M[2, :3] = s, u, -f; M[:3, 3] = -M[:3, :3] @ eye; return M

def persp(fov, n=0.1, fa=10):
    t = np.tan(np.radians(fov) / 2); P = np.zeros((4, 4))
    P[0, 0] = P[1, 1] = 1 / t; P[2, 2] = (fa + n) / (n - fa); P[2, 3] = 2 * fa * n / (n - fa); P[3, 2] = -1; return P

def render(V, F, az, el, ctr, dist, fov, res=700):
    a, e = np.radians(az), np.radians(el)
    eye = np.asarray(ctr) + dist * np.array([np.sin(a) * np.cos(e), np.sin(e), np.cos(a) * np.cos(e)])
    mvp = persp(fov) @ lookat(eye, ctr)
    vt = torch.tensor(V, dtype=torch.float32, device=DEV); ft = torch.tensor(F, dtype=torch.int32, device=DEV)
    clip = torch.cat([vt, torch.ones(len(vt), 1, device=DEV)], 1) @ torch.tensor(mvp, dtype=torch.float32, device=DEV).T
    rast, _ = dr.rasterize(ctx, clip[None].contiguous(), ft, (res, res))
    fid = rast[0, ..., 3].long()
    fn = torch.cross(vt[ft[:, 1].long()] - vt[ft[:, 0].long()], vt[ft[:, 2].long()] - vt[ft[:, 0].long()], dim=1)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-12)
    ldir = torch.tensor(eye - ctr, dtype=torch.float32, device=DEV); ldir /= ldir.norm()
    l2 = torch.tensor([0.3, 0.8, 0.5], device=DEV); l2 /= l2.norm()
    sh = 0.25 + 0.55 * (fn @ ldir).abs() + 0.2 * (fn @ l2).abs()
    img = torch.ones(res, res, device=DEV)
    m = fid > 0; img[m] = sh[fid[m] - 1]
    return (img.cpu().numpy()[::-1] * 255).astype(np.uint8)  # nvdiffrast row0 = bottom

ROWS = [("front", dict(az=180, el=10, ctr=(-0.3, 0.1, -0.3), dist=5.6, fov=40)),
        ("back", dict(az=20, el=15, ctr=(-0.3, 0.1, -0.3), dist=5.6, fov=40)),
        ("head closeup", dict(az=180, el=15, ctr=(-0.1, 1.35, -0.7), dist=2.2, fov=35)),
        ("back closeup", dict(az=10, el=25, ctr=(-0.3, 0.3, 0.3), dist=2.2, fov=35))]
_rows = os.environ.get("ROWS")           # optional comma list of row names to keep
if _rows: ROWS = [r for r in ROWS if r[0] in _rows.split(",")]
out = sys.argv[1]; items = [a.split("=", 1) for a in sys.argv[2:]]
res = 700; W = res * len(items); H = res * len(ROWS) + 60
canvas = Image.new("L", (W, H), 255); d = ImageDraw.Draw(canvas)
try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 30)
except Exception: font = ImageFont.load_default()
meshes = [(lab, load(p)) for lab, p in items]
for j, (lab, (V, F)) in enumerate(meshes):
    d.text((j * res + 10, 12), f"{lab}  ({len(F)/1000:.1f}k f)", fill=0, font=font)
    for i, (rn, kw) in enumerate(ROWS):
        canvas.paste(Image.fromarray(render(V, F, **kw)), (j * res, 60 + i * res))
for i, (rn, _) in enumerate(ROWS): d.text((10, 60 + i * res + 8), rn, fill=90, font=font)
canvas.save(out); print("saved", out, canvas.size)
