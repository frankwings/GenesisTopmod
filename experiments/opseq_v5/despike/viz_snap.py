"""Per-step video frames: mosaic of flat-shaded renders from ALL training cameras.
Global frame counter persisted in SNAPSHOT_DIR/.counter so consecutive stages
(run_64v -> phase4 -> phase5 ...) append to one continuous frame sequence.
Enabled only when env SNAPSHOT_DIR is set; SNAPSHOT_EVERY (default 1) thins steps.
"""
import os, numpy as np, torch
import nvdiffrast.torch as dr
from PIL import Image, ImageDraw, ImageFont

SNAPSHOT_DIR = os.environ.get("SNAPSHOT_DIR", "")
SNAPSHOT_EVERY = int(os.environ.get("SNAPSHOT_EVERY", "1"))
_font = None

def enabled():
    return bool(SNAPSHOT_DIR)

def _next_index():
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)
    p = os.path.join(SNAPSHOT_DIR, ".counter")
    n = int(open(p).read()) if os.path.exists(p) else 0
    open(p, "w").write(str(n + 1)); return n

def snap(ctx, mvps, V, F, title, step=None, res=192, cols=8, hold=1):
    """Write one (or `hold` identical) frames. `step`=None bypasses SNAPSHOT_EVERY."""
    global _font
    if not SNAPSHOT_DIR: return
    if step is not None and step % SNAPSHOT_EVERY != 0: return
    dev = "cuda"
    V = np.asarray(V, np.float64); F = np.asarray(F, np.int64)
    vt = torch.tensor(V, dtype=torch.float32, device=dev); ft = torch.tensor(F, dtype=torch.int32, device=dev)
    fn = torch.cross(vt[ft[:, 1].long()] - vt[ft[:, 0].long()], vt[ft[:, 2].long()] - vt[ft[:, 0].long()], dim=1)
    fn = fn / (fn.norm(dim=1, keepdim=True) + 1e-12)
    l1 = torch.tensor([0.3, 0.8, 0.5], device=dev); l1 /= l1.norm()
    l2 = torch.tensor([-0.6, 0.2, -0.8], device=dev); l2 /= l2.norm()
    sh = 0.3 + 0.45 * (fn @ l1).abs() + 0.25 * (fn @ l2).abs()
    hom = torch.cat([vt, torch.ones(len(vt), 1, device=dev)], 1)
    n = len(mvps); rows = (n + cols - 1) // cols
    canvas = Image.new("L", (cols * res, rows * res + 36), 255)
    for i in range(n):
        m = torch.as_tensor(mvps[i]).float().to(dev)
        rast, _ = dr.rasterize(ctx, (hom @ m.T)[None].contiguous(), ft, (res, res))
        fid = rast[0, ..., 3].long(); img = torch.ones(res, res, device=dev); msk = fid > 0
        img[msk] = sh[fid[msk] - 1]
        canvas.paste(Image.fromarray((img.cpu().numpy()[::-1] * 255).astype(np.uint8)), ((i % cols) * res, 36 + (i // cols) * res))
    if _font is None:
        try: _font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        except Exception: _font = ImageFont.load_default()
    ImageDraw.Draw(canvas).text((10, 6), f"{title}   V={len(V)} F={len(F)}   {n} training views", fill=0, font=_font)
    for _ in range(hold):
        canvas.save(os.path.join(SNAPSHOT_DIR, f"frame_{_next_index():06d}.png"))
