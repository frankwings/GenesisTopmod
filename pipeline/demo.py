"""
Phase 2 end-to-end demo: Topology-First Geometry Optimization.

For each demo case:
    1. Build a reference mesh with known topology (sphere or torus) using trimesh.
    2. Render reference from N camera views → target silhouette images.
    3. Build a topology-correct seed mesh via build_topology().
    4. Optimize vertex positions to match target silhouettes.
    5. Save:
       - reference.obj  (ground-truth geometry)
       - initial.obj    (seed topology, wrong geometry)
       - final.obj      (optimized geometry)
       - comparison.png (side-by-side silhouette grid)
       - loss_curve.png (loss history)

Usage
-----
    python pipeline/demo.py [--device cuda] [--steps 500] [--views 8] [--res 256]
    python pipeline/demo.py --case sphere
    python pipeline/demo.py --case torus
    python pipeline/demo.py --case both   (default)
"""

from __future__ import annotations
import argparse
import math
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh

# ── project imports ───────────────────────────────────────────────────────────
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from topmod.io import to_obj
from topmod.primitives import _build_mesh

from pipeline.topology_builder import build_topology
from pipeline.cameras import orbit_cameras, transform_to_clip
from pipeline.geometry_optimizer import optimize, render_batch


# ── output dir ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("/tmp/topmod_phase2")


# ── reference mesh helpers ────────────────────────────────────────────────────

def _trimesh_to_tensors(
    tm: trimesh.Trimesh,
    scale: float,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a trimesh.Trimesh to GPU tensors, centred and scaled."""
    verts_np = np.array(tm.vertices, dtype=np.float32)
    faces_np = np.array(tm.faces,    dtype=np.int32)

    # Centre at origin
    verts_np -= verts_np.mean(axis=0, keepdims=True)

    # Normalise to fit in [-scale, scale]
    max_extent = np.abs(verts_np).max()
    if max_extent > 1e-6:
        verts_np = verts_np / max_extent * scale

    verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    faces = torch.tensor(faces_np, dtype=torch.int32,   device=device)
    return verts, faces


def make_reference_sphere(scale: float = 0.85, device: str = "cuda"):
    """Unit sphere via trimesh icosphere subdivision."""
    tm = trimesh.creation.icosphere(subdivisions=4)
    return _trimesh_to_tensors(tm, scale, device)


def make_reference_torus(
    major_radius: float = 1.0,
    minor_radius: float = 0.30,
    scale:        float = 0.80,
    device:       str   = "cuda",
):
    """Torus (major=1.0, minor=0.30) via trimesh."""
    tm = trimesh.creation.torus(
        major_radius=major_radius,
        minor_radius=minor_radius,
        major_sections=48,
        minor_sections=24,
    )
    return _trimesh_to_tensors(tm, scale, device)


# ── render reference silhouettes ──────────────────────────────────────────────

@torch.no_grad()
def render_reference(
    ctx:     dr.RasterizeCudaContext,
    verts:   torch.Tensor,   # [V, 3]
    faces:   torch.Tensor,   # [F, 3]
    mvps:    torch.Tensor,   # [N, 4, 4]
    res:     int,
) -> torch.Tensor:           # [N, H, W, 1]
    """Render binary silhouette targets from a reference mesh (no grad)."""
    from pipeline.geometry_optimizer import render_silhouette
    N = mvps.shape[0]
    imgs = []
    for i in range(N):
        sil = render_silhouette(ctx, verts, faces, mvps[i], (res, res))
        imgs.append(sil)
    # Binarize: rasterized pixels are exactly 1 inside, 0 outside (no antialias
    # on target) — we keep soft values from antialias, they're already ~0/1.
    target = torch.cat(imgs, dim=0)   # [N, H, W, 1]
    return target.clamp(0, 1)


# ── save utilities ────────────────────────────────────────────────────────────

def _tensors_to_dlfl(verts_t: torch.Tensor, faces_t: torch.Tensor):
    """Convert GPU tensors back to a DLFLMesh for OBJ export."""
    verts_np = verts_t.cpu().numpy()
    faces_np = faces_t.cpu().numpy().tolist()
    positions = [tuple(v) for v in verts_np]
    return _build_mesh(positions, faces_np)


def save_obj_from_tensors(
    verts_t: torch.Tensor,
    faces_t: torch.Tensor,
    path: str,
) -> None:
    """Export (verts, faces) tensors as an OBJ file."""
    mesh = _tensors_to_dlfl(verts_t, faces_t)
    to_obj(mesh, path)


def save_comparison_png(
    target:  torch.Tensor,   # [N, H, W, 1]
    initial: torch.Tensor,   # [N, H, W, 1]
    final:   torch.Tensor,   # [N, H, W, 1]
    path:    str,
    max_views: int = 8,
) -> None:
    """Save a side-by-side PNG: target | initial | final (rows = views)."""
    from PIL import Image

    N   = min(target.shape[0], max_views)
    H,W = target.shape[1], target.shape[2]

    # Convert [N, H, W, 1] → [N, H, W, 3] → numpy uint8
    def to_rgb(t):
        arr = t[:N, :, :, 0].cpu().clamp(0, 1).numpy()
        return (arr * 255).astype(np.uint8)

    tgt_arr = to_rgb(target)
    ini_arr = to_rgb(initial)
    fin_arr = to_rgb(final)

    # Lay out: each row = one view, columns = [target | initial | final]
    gap   = 4
    cols  = 3
    full_w = W * cols + gap * (cols - 1)
    full_h = H * N    + gap * (N  - 1)

    canvas = np.ones((full_h, full_w), dtype=np.uint8) * 180   # grey bg

    for row in range(N):
        y = row * (H + gap)
        for col, arr in enumerate([tgt_arr, ini_arr, fin_arr]):
            x = col * (W + gap)
            canvas[y : y + H, x : x + W] = arr[row]

    img = Image.fromarray(canvas, mode="L")
    img.save(path)
    print(f"  saved {path}")


def save_loss_curve(history: list, path: str) -> None:
    """Save a loss curve PNG using matplotlib (if available)."""
    try:
        import matplotlib.pyplot as plt
        steps = [h["step"]     for h in history]
        total = [h["loss"]     for h in history]
        sil   = [h["sil_loss"] for h in history]
        reg   = [h["reg_loss"] for h in history]

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.semilogy(steps, total, label="total",  lw=2)
        ax.semilogy(steps, sil,   label="sil",    lw=1.5, ls="--")
        ax.semilogy(steps, reg,   label="reg",    lw=1.5, ls=":")
        ax.set_xlabel("step");  ax.set_ylabel("loss (log)")
        ax.legend();  ax.grid(True, which="both", alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=120)
        plt.close(fig)
        print(f"  saved {path}")
    except ImportError:
        print("  matplotlib not found — skipping loss curve")


# ── single demo case ──────────────────────────────────────────────────────────

def run_demo_case(
    name:        str,
    genus:       int,
    ref_verts:   torch.Tensor,
    ref_faces:   torch.Tensor,
    ctx:         dr.RasterizeCudaContext,
    mvps:        torch.Tensor,
    num_steps:   int,
    res:         int,
    output_dir:  Path,
    subdivisions: int = 2,
) -> dict:
    """Run one topology+geometry demo case.  Returns summary dict."""
    print(f"\n{'='*60}")
    print(f"  Demo: {name}  (genus={genus}, steps={num_steps}, res={res}²)")
    print(f"{'='*60}")
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)

    device = ref_verts.device

    # ── Render reference silhouettes ──────────────────────────────────
    print("\n[1/4] Rendering reference silhouettes...")
    target_images = render_reference(ctx, ref_verts, ref_faces, mvps, res)
    print(f"      target_images: {target_images.shape}  "
          f"coverage={target_images.mean().item():.3f}")

    # ── Build topology seed mesh ──────────────────────────────────────
    print(f"\n[2/4] Building genus-{genus} seed mesh (subdivisions={subdivisions})...")
    t0 = time.perf_counter()
    init_verts, init_faces = build_topology(
        genus=genus, boundaries=0, subdivisions=subdivisions,
        scale=0.6, device=device
    )
    print(f"      seed: V={init_verts.shape[0]} F={init_faces.shape[0]}  "
          f"({time.perf_counter()-t0:.2f}s)")

    # Save initial OBJ
    save_obj_from_tensors(ref_verts,  ref_faces,  str(case_dir / "reference.obj"))
    save_obj_from_tensors(init_verts, init_faces, str(case_dir / "initial.obj"))

    # Render initial silhouettes for comparison image
    with torch.no_grad():
        init_images = render_reference(ctx, init_verts, init_faces, mvps, res)

    # ── Optimise ──────────────────────────────────────────────────────
    print(f"\n[3/4] Optimising vertex positions ({num_steps} steps)...")
    t0 = time.perf_counter()
    final_verts, history = optimize(
        ctx=ctx,
        verts_init=init_verts,
        faces=init_faces,
        target_images=target_images,
        mvps=mvps,
        num_steps=num_steps,
        lr=5e-3,
        lambda_lap=0.05,
        lambda_edge=0.01,
        lambda_normal=0.0,
        resolution=(res, res),
        log_every=max(1, num_steps // 10),
        scheduler=True,
    )
    elapsed = time.perf_counter() - t0
    print(f"      optimisation done in {elapsed:.1f}s")

    initial_loss = history[0]["sil_loss"]
    final_loss   = history[-1]["sil_loss"]
    converged    = final_loss < initial_loss * 0.70   # 30 % improvement
    print(f"      sil_loss: {initial_loss:.4f} → {final_loss:.4f}  "
          f"({'✓ converged' if converged else '△ partial'})")

    # ── Save outputs ──────────────────────────────────────────────────
    print(f"\n[4/4] Saving outputs to {case_dir}/")
    save_obj_from_tensors(final_verts, init_faces, str(case_dir / "final.obj"))

    with torch.no_grad():
        final_images = render_reference(ctx, final_verts, init_faces, mvps, res)

    save_comparison_png(
        target_images, init_images, final_images,
        str(case_dir / "comparison.png"),
        max_views=min(6, mvps.shape[0]),
    )
    save_loss_curve(history, str(case_dir / "loss_curve.png"))

    return {
        "name":          name,
        "genus":         genus,
        "V":             final_verts.shape[0],
        "F":             init_faces.shape[0],
        "initial_loss":  initial_loss,
        "final_loss":    final_loss,
        "improvement":   (initial_loss - final_loss) / max(initial_loss, 1e-8),
        "converged":     converged,
        "elapsed_s":     elapsed,
    }


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GenesisTopmod Phase 2 Demo")
    parser.add_argument("--case",   default="both",  choices=["sphere", "torus", "both"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps",  default=500, type=int,  help="optimisation steps")
    parser.add_argument("--views",  default=8,   type=int,  help="camera views")
    parser.add_argument("--res",    default=256, type=int,  help="render resolution")
    parser.add_argument("--subs",   default=2,   type=int,  help="CC subdivisions")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = args.device

    print(f"GenesisTopmod Phase 2 Demo")
    print(f"  device={device}  steps={args.steps}  views={args.views}  res={args.res}")
    print(f"  output → {OUTPUT_DIR}/")

    # ── nvdiffrast context ────────────────────────────────────────────
    ctx = dr.RasterizeCudaContext()

    # ── Shared camera rig ─────────────────────────────────────────────
    mvps, eyes = orbit_cameras(
        n=args.views, elevation_deg=25.0, radius=3.0,
        fov_deg=40.0, device=device
    )
    print(f"\nCamera rig: {args.views} views, elevation=25°, radius=3.0")

    summaries = []
    cases = []
    if args.case in ("sphere", "both"):
        ref_v, ref_f = make_reference_sphere(scale=0.85, device=device)
        cases.append(("sphere_genus0", 0, ref_v, ref_f))
    if args.case in ("torus", "both"):
        ref_v, ref_f = make_reference_torus(scale=0.80, device=device)
        cases.append(("torus_genus1", 1, ref_v, ref_f))

    for name, genus, ref_v, ref_f in cases:
        summary = run_demo_case(
            name=name, genus=genus,
            ref_verts=ref_v, ref_faces=ref_f,
            ctx=ctx, mvps=mvps,
            num_steps=args.steps,
            res=args.res,
            output_dir=OUTPUT_DIR,
            subdivisions=args.subs,
        )
        summaries.append(summary)

    # ── Print summary ─────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for s in summaries:
        status = "✓ CONVERGED" if s["converged"] else "△ PARTIAL"
        print(
            f"  {s['name']:20s} | "
            f"V={s['V']:5d} F={s['F']:5d} | "
            f"sil: {s['initial_loss']:.4f}→{s['final_loss']:.4f} "
            f"({s['improvement']*100:.1f}% impr) | "
            f"{s['elapsed_s']:.1f}s | {status}"
        )
    print(f"\nOutputs written to {OUTPUT_DIR}/")

    all_converged = all(s["converged"] for s in summaries)
    sys.exit(0 if all_converged else 1)


if __name__ == "__main__":
    main()
