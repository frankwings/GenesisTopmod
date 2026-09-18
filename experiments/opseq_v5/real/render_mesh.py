#!/usr/bin/env python3
"""render_mesh.py <mesh.npz|obj> <out_prefix> [--turn N]: shaded turntable renders with nvdiffrast (no GT needed).
Writes <out_prefix>_grid.png (8 views) and, with --turn N, <out_prefix>_turn.mp4 (N frames)."""
import sys, os, math, argparse, subprocess, numpy as np, torch, cv2
sys.path[:0] = ["/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5", "/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/despike", "/home/kingy/Projects/Genesis/GenesisTopmod"]
import nvdiffrast.torch as dr
from pipeline.cameras import perspective, look_at, transform_to_clip
ap = argparse.ArgumentParser(); ap.add_argument("mesh"); ap.add_argument("out"); ap.add_argument("--turn", type=int, default=0); ap.add_argument("--res", type=int, default=560); ap.add_argument("--up", default="0,1,0"); ap.add_argument("--colors", default="", help="per-vertex RGB in npz key 'colors'"); ap.add_argument("--no-colors", action="store_true", help="ignore vertex colours even if present"); a = ap.parse_args()
if a.mesh.endswith(".npz"):
    z = np.load(a.mesh); V, Fc = z["verts"].astype(np.float32), z["tris"].astype(np.int32)
    vc_np = z["colors"].astype(np.float32) if (a.colors == "" and "colors" in z) or a.colors else None
    if a.colors and a.colors != a.mesh:
        zc = np.load(a.colors); vc_np = zc["colors"].astype(np.float32)
    if a.no_colors: vc_np = None   # geometry-only mode
else:
    import trimesh; m = trimesh.load(a.mesh, force="mesh"); V, Fc = m.vertices.astype(np.float32), m.faces.astype(np.int32); vc_np = None
V = V - V.mean(0); V /= np.abs(V).max()
up = np.array([float(x) for x in a.up.split(",")]); up /= np.linalg.norm(up)
dev = "cuda"; ctx = dr.RasterizeCudaContext(); vt = torch.tensor(V, device=dev); ft = torch.tensor(Fc, device=dev)
fn = np.cross(V[Fc[:, 1]] - V[Fc[:, 0]], V[Fc[:, 2]] - V[Fc[:, 0]]); vn = np.zeros_like(V)
for k in range(3): np.add.at(vn, Fc[:, k], fn)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-9; vnt = torch.tensor(vn, device=dev)
vct = torch.tensor(vc_np, device=dev) if vc_np is not None else None   # per-vertex colours [V,3]
proj = perspective(fov_deg=40.0, aspect=1.0, near=0.1, far=20.0, device=dev)
e1 = np.cross(up, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
def render(az_deg, el_deg, R=3.2):
    a_, e_ = math.radians(az_deg), math.radians(el_deg)
    eye = R * (math.cos(e_) * (math.cos(a_) * e1 + math.sin(a_) * e2) + math.sin(e_) * up)
    view = look_at(tuple(map(float, eye)), center=(0, 0, 0), up=tuple(map(float, up)), device=dev); mvp = proj @ view
    pos = transform_to_clip(vt, mvp); rast, _ = dr.rasterize(ctx, pos, ft, resolution=[a.res, a.res])
    n_cam = (vnt @ view[:3, :3].T).unsqueeze(0).contiguous(); ni, _ = dr.interpolate(n_cam, rast, ft)
    ni = ni[0] / (ni[0].norm(dim=-1, keepdim=True) + 1e-9); fg = rast[0, :, :, 3] > 0
    key = ni[..., 2].abs(); fill = (ni @ torch.tensor([0.5, 0.6, 0.62], device=dev)).clamp(min=0)
    shade = (0.18 + 0.62 * key + 0.35 * fill).clamp(0, 1)
    if vct is not None:
        # flat-shaded vertex colours x soft key light
        vc_img, _ = dr.interpolate(vct.unsqueeze(0).contiguous(), rast, ft)
        col = vc_img[0].clamp(0, 1) * (0.35 + 0.65 * key.unsqueeze(-1))
    else:
        col = torch.stack([shade * 0.62, shade * 0.78, shade * 0.95], -1)
    img = torch.where(fg[..., None], col, torch.ones_like(col))
    img = dr.antialias(img.unsqueeze(0).contiguous(), rast, pos, ft)[0]
    return (img.flip(0).cpu().numpy() * 255).astype(np.uint8)   # nvdiffrast row0 = bottom -> flip for image
tiles = [render(az, el) for az, el in [(0, 15), (45, 15), (90, 15), (135, 15), (180, 15), (225, 15), (270, 15), (0, 70)]]
grid = np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0)
cv2.putText(grid, f"{os.path.basename(a.mesh)}  V={len(V)} F={len(Fc)}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (40, 40, 40), 2)
cv2.imwrite(a.out + "_grid.png", cv2.cvtColor(grid, cv2.COLOR_RGB2BGR)); print("grid", a.out + "_grid.png")
if a.turn:
    d = a.out + "_frames"; os.makedirs(d, exist_ok=True)
    for i in range(a.turn): cv2.imwrite(f"{d}/{i:03d}.png", cv2.cvtColor(render(360.0 * i / a.turn, 18), cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{d}/%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out + "_turn.mp4"], check=True); print("mp4", a.out + "_turn.mp4")
