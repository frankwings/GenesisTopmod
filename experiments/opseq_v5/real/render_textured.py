#!/usr/bin/env python3
"""render_textured.py <mesh.npz with verts,tris,uvs,uv_idx> <texture.png> <out_prefix> [--turn N] [--up x,y,z]
Free-viewpoint renders of a UV-textured mesh (nvdiffrast dr.texture): <out>_grid.png (8 views) and <out>_turn.mp4."""
import sys, os, math, argparse, subprocess, numpy as np, torch, cv2
sys.path[:0] = ["/home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5", "/home/kingy/Projects/Genesis/GenesisTopmod"]
import nvdiffrast.torch as dr
from pipeline.cameras import perspective, look_at
ap = argparse.ArgumentParser(); ap.add_argument("mesh"); ap.add_argument("tex"); ap.add_argument("out"); ap.add_argument("--turn", type=int, default=0)
ap.add_argument("--res", type=int, default=560); ap.add_argument("--up", default="0,1,0"); ap.add_argument("--flip-v", type=int, default=0); ap.add_argument("--tex-down", type=int, default=1, help="box-filter the texture by this factor before rendering (texels finer than the training pixels are only constrained on average)"); a = ap.parse_args()
z = np.load(a.mesh); V = z["verts"].astype(np.float32); F = z["tris"].astype(np.int32); uv = z["uvs"].astype(np.float32); uvi = z["uv_idx"].astype(np.int32)
V = V - V.mean(0); V /= np.abs(V).max(); up = np.array([float(x) for x in a.up.split(",")]); up /= np.linalg.norm(up)
T = cv2.cvtColor(cv2.imread(a.tex), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
if a.flip_v: T = T[::-1].copy()
if a.tex_down > 1: T = cv2.resize(T, (T.shape[1] // a.tex_down, T.shape[0] // a.tex_down), interpolation=cv2.INTER_AREA)
dev = "cuda"; ctx = dr.RasterizeCudaContext(); vt = torch.tensor(V, device=dev); ft = torch.tensor(F, device=dev); uvt = torch.tensor(uv, device=dev)[None]; uit = torch.tensor(uvi, device=dev); Tt = torch.tensor(T, device=dev)[None].contiguous()
proj = perspective(fov_deg=40.0, aspect=1.0, near=0.1, far=20.0, device=dev); e1 = np.cross(up, [1, 0, 0]); e1 /= np.linalg.norm(e1); e2 = np.cross(up, e1)
def render(az, el, R=3.2):
    a_, e_ = math.radians(az), math.radians(el); eye = R * (math.cos(e_) * (math.cos(a_) * e1 + math.sin(a_) * e2) + math.sin(e_) * up)
    mvp = proj @ look_at(tuple(map(float, eye)), center=(0, 0, 0), up=tuple(map(float, up)), device=dev)
    pos = (torch.cat([vt, torch.ones(len(V), 1, device=dev)], 1) @ mvp.T)[None].contiguous()
    rast, rdb = dr.rasterize(ctx, pos, ft, resolution=[a.res, a.res]); tc, tcd = dr.interpolate(uvt, rast, uit, rast_db=rdb, diff_attrs="all")
    col = dr.texture(Tt, tc, tcd, filter_mode="linear-mipmap-linear"); fg = rast[..., 3:4] > 0
    img = dr.antialias(torch.where(fg, col, torch.ones_like(col)).contiguous(), rast, pos, ft)[0]
    return (img.flip(0).clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)
tiles = [render(az, el) for az, el in [(0, 15), (45, 15), (90, 15), (135, 15), (180, 15), (225, 15), (270, 15), (0, 70)]]
cv2.imwrite(a.out + "_grid.png", cv2.cvtColor(np.concatenate([np.concatenate(tiles[:4], 1), np.concatenate(tiles[4:], 1)], 0), cv2.COLOR_RGB2BGR)); print("grid", a.out + "_grid.png")
if a.turn:
    d = a.out + "_frames"; os.makedirs(d, exist_ok=True)
    for i in range(a.turn): cv2.imwrite(f"{d}/{i:03d}.png", cv2.cvtColor(render(360.0 * i / a.turn, 18), cv2.COLOR_RGB2BGR))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", "24", "-i", f"{d}/%03d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", a.out + "_turn.mp4"], check=True); print("mp4", a.out + "_turn.mp4")
