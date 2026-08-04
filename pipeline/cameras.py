"""
Camera utilities for the geometry optimization pipeline.

Functions
---------
orbit_cameras(n, elevation_deg, radius, fov_deg) -> (mvps, eyes)
    Generate N MVP matrices on a circular orbit around the origin.

perspective(fov_deg, aspect, near, far) -> [4, 4] float32 tensor
look_at(eye, center, up)               -> [4, 4] float32 tensor
"""

from __future__ import annotations
import math
from typing import List, Tuple

import torch


# ── perspective projection ────────────────────────────────────────────────────

def perspective(
    fov_deg: float = 45.0,
    aspect:  float = 1.0,
    near:    float = 0.1,
    far:     float = 10.0,
    device:  str   = "cuda",
) -> torch.Tensor:
    """
    Right-handed OpenGL perspective projection matrix.

    NDC x, y ∈ [-1, 1];  z ∈ [-1, 1] (before clip).
    """
    fov = math.radians(fov_deg)
    f   = 1.0 / math.tan(fov / 2.0)
    m   = torch.zeros(4, 4, dtype=torch.float32, device=device)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2.0 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


# ── look-at view matrix ───────────────────────────────────────────────────────

def look_at(
    eye:    Tuple[float, float, float],
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    up:     Tuple[float, float, float] = (0.0, 1.0, 0.0),
    device: str = "cuda",
) -> torch.Tensor:
    """
    Right-handed look-at view matrix.

    Camera looks from *eye* toward *center*.  In clip space, camera faces -Z.
    """
    e = torch.tensor(eye,    dtype=torch.float32)
    c = torch.tensor(center, dtype=torch.float32)
    u = torch.tensor(up,     dtype=torch.float32)

    f = c - e; f = f / f.norm()               # forward
    r = torch.linalg.cross(f, u); r = r / r.norm()   # right
    u2 = torch.linalg.cross(r, f)             # true up

    view = torch.eye(4, dtype=torch.float32)
    view[0, :3] =  r;  view[0, 3] = -torch.dot(r,  e)
    view[1, :3] =  u2; view[1, 3] = -torch.dot(u2, e)
    view[2, :3] = -f;  view[2, 3] =  torch.dot(f,  e)
    return view.to(device)


# ── orbital camera rig ────────────────────────────────────────────────────────

def orbit_cameras(
    n:             int   = 8,
    elevation_deg: float = 20.0,
    radius:        float = 3.0,
    fov_deg:       float = 45.0,
    near:          float = 0.1,
    far:           float = 10.0,
    device:        str   = "cuda",
) -> Tuple[torch.Tensor, List[Tuple[float, float, float]]]:
    """
    Generate *n* MVP matrices equally spaced on a horizontal orbit at the
    given *elevation_deg* above the equator.

    Returns
    -------
    mvps : [N, 4, 4] float32 tensor on *device*
    eyes : list of N (x, y, z) camera positions
    """
    proj = perspective(fov_deg=fov_deg, near=near, far=far, device=device)

    elev_rad = math.radians(elevation_deg)
    y   = radius * math.sin(elev_rad)
    r_h = radius * math.cos(elev_rad)   # horizontal radius

    mvps: List[torch.Tensor] = []
    eyes: List[Tuple[float, float, float]] = []

    for i in range(n):
        az = 2.0 * math.pi * i / n
        x  = r_h * math.cos(az)
        z  = r_h * math.sin(az)
        eye = (x, y, z)
        eyes.append(eye)

        view = look_at(eye, center=(0, 0, 0), up=(0, 1, 0), device=device)
        mvps.append(proj @ view)

    return torch.stack(mvps, dim=0), eyes   # [N, 4, 4]


# ── transform helper ──────────────────────────────────────────────────────────

def transform_to_clip(
    verts: torch.Tensor,   # [V, 3]
    mvp:   torch.Tensor,   # [4, 4]
) -> torch.Tensor:
    """
    Transform [V, 3] vertices to clip-space [1, V, 4] (homogeneous).

    Result is guaranteed to be contiguous (required by nvdiffrast rasterize).
    """
    V = verts.shape[0]
    ones = torch.ones(V, 1, dtype=verts.dtype, device=verts.device)
    verts_h = torch.cat([verts, ones], dim=-1)     # [V, 4]
    pos_clip = (mvp @ verts_h.T).T.unsqueeze(0)   # [1, V, 4]
    return pos_clip.contiguous()
