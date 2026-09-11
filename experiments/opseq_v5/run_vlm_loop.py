#!/usr/bin/env python3
"""
run_vlm_loop.py — VLM-driven TopMod operator selector loop (Phase 1).

Architecture
------------
  L1: Gemini (VLM) — given a 6-view triptych image, picks region + operator
  L2: Python      — grounds the region to face IDs, executes TopMod operator

Loop per optimisation step:
  1. Render 6 views → compute IoU
  2. Plateau detection → trigger VLM operator selection
  3. VLM: encode triptych, send image+prompt → parse region number
  4. Ground region → cluster face IDs + extrude_dir
  5. Execute topmod_extrude_cluster → new mesh
  6. _rebuild_opt (cold-start Adam) → continue optimisation
  7. Log step to trajectory.jsonl

Usage
-----
    # Quick local test (300 steps, no bg_task):
    python3 run_vlm_loop.py --steps 300 --out_dir eval_out/vlm_run

    # Full 800-step run (>5 min → must be launched via bg_task):
    python3 -m framework.bg_task launch --id vlm_loop_run1 --timeout 3600 \
        --cwd /home/kingy/Foundation/ZenithLoom \
        -- python3 /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5/run_vlm_loop.py \
           --steps 800 --out_dir eval_out/vlm_run1

Outputs
-------
    <out_dir>/trajectory.jsonl          — per-injection JSONL records
    <out_dir>/step_XXXX_state.png       — 6-view triptych at each injection
    <out_dir>/final_render.png          — 6-view grid at end of run
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvdiffrast.torch as dr

from pipeline.cameras            import orbit_cameras, transform_to_clip
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)
from eval_v5          import adaptive_remesh
from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH

from eval_extrude_v3 import (
    make_6_cameras,
    render_sil_and_depth,
    render_views_n,
    render_depths_n,
    compute_iou_n,
    depth_loss_masked,
    vote_faces_multiview_n,
    dlfl_boundary_stats,
    topmod_extrude_cluster,
    split_new_long_edges,
)
from eval_extrude_inject import (
    select_face_cluster,
    compute_missing_error_maps,
    detect_topology_bottleneck,
    largest_cc_fraction,
)

from vlm_client        import VLMClient
from vlm_state_encoder import encode_state
from operator_grounding import build_candidates, ground_region
from operator_tools     import execute_operator
from trajectory_logger  import TrajectoryLogger


# ── hyper-parameters ──────────────────────────────────────────────────────────
IMG_RES          = 256
N_VIEWS          = 6
LR               = 3e-3
LR_MIN           = 3e-5
W_LAP            = 0.10
W_LAP_BOOST      = 0.30
W_EDGE           = 0.01
W_DEPTH          = 0.35
PLATEAU_STEPS    = 20
PLATEAU_EPS      = 0.005
EVAL_INTERVAL    = 5
WARMUP_STEPS     = 20
LAP_WARMUP_STEPS = 30
ERROR_CC_THRESH  = 0.02
EXTRUDE_DIST     = 0.05
MIN_STEP_FOR_INJ = 60
MAX_INJECTIONS   = 5
N_CANDIDATES     = 3
CLUSTER_SIZE     = 8


# ── VLM prompt ────────────────────────────────────────────────────────────────

_VLM_SYSTEM_PROMPT = """You are a 3D mesh topology advisor.
You will see a 6-row image. Each row shows one camera view:
  Column 1 (left):   current rendered silhouette (white=mesh, black=background)
  Column 2 (centre): ground-truth silhouette
  Column 3 (right):  error heatmap — red pixels = geometry MISSING in current render;
                     numbered coloured circles = candidate regions to extrude

Your task: identify which numbered region (1, 2, or 3) most consistently appears
in the red-pixel areas across the most views.

Reply with EXACTLY one line in this format:
REGION: <number>
REASON: <one sentence>

Do not add anything else."""


def _parse_vlm_reply(reply: str) -> int:
    """Extract region number from VLM reply.  Returns 1 on parse failure."""
    m = re.search(r"REGION\s*:\s*(\d+)", reply, re.IGNORECASE)
    if m:
        return int(m.group(1))
    # Fallback: first digit found
    m2 = re.search(r"\b([1-9])\b", reply)
    if m2:
        return int(m2.group(1))
    return 1


# ── optimisation utilities ────────────────────────────────────────────────────

def _rebuild_opt(
    v_np:       np.ndarray,
    t_np:       np.ndarray,
    cur_step:   int,
    total_steps: int,
    device:     str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.optim.Optimizer, object]:
    """Cold-start Adam + cosine scheduler on new mesh."""
    vt = torch.tensor(v_np, dtype=torch.float32, device=device).requires_grad_(True)
    ft = torch.tensor(t_np, dtype=torch.int32,   device=device)
    opt  = torch.optim.Adam([vt], lr=LR)
    rem  = max(total_steps - cur_step - 1, 1)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=rem, eta_min=LR_MIN)
    return vt, ft, opt, sched


def _plateau_check(
    iou_history: List[float],
    window:      int,
    eps:         float,
) -> bool:
    """Return True if IoU improved less than eps over the last `window` entries."""
    if len(iou_history) < window:
        return False
    recent = iou_history[-window:]
    return recent[-1] - recent[0] < eps


# ── main loop ────────────────────────────────────────────────────────────────

def run_vlm_loop(
    device:      str,
    total_steps: int,
    out_dir:     str,
    dry_run:     bool = False,
    shape:       str  = "bunny",
    max_injections: int = MAX_INJECTIONS,
) -> Dict:
    """Run the VLM operator loop on a target shape.

    Parameters
    ----------
    device : str
    total_steps : int
    out_dir : str
    dry_run : bool
        If True, skip VLM calls and always pick region 1 (for CI / unit testing).
    shape : str
        Shape name: "bunny", "cow", "airplane", "torus", etc.

    Returns
    -------
    dict with keys: final_iou, n_injections, trajectory_path
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # ── load mesh ─────────────────────────────────────────────────────────
    _SAMPLE_DIR = os.path.dirname(BUNNY_PATH)
    if shape == "bunny":
        obj_path = BUNNY_PATH
    else:
        obj_path = os.path.join(_SAMPLE_DIR, f"{shape}.obj")
    if not os.path.exists(obj_path):
        raise FileNotFoundError(f"Shape file not found: {obj_path}")
    print(f"  Loading shape '{shape}' from {obj_path}")
    verts_init, tris_init = load_obj(obj_path)
    verts_init = normalize_to_range(verts_init)

    # ── cameras ───────────────────────────────────────────────────────────
    ctx  = dr.RasterizeCudaContext()
    mvps, eyes = make_6_cameras(device=device)

    # ── render GT silhouettes + depths ────────────────────────────────────
    verts_gt = torch.tensor(verts_init, dtype=torch.float32, device=device)
    faces_gt = torch.tensor(tris_init,  dtype=torch.int32,   device=device)
    gt_uint8 = []
    gt_depths = []
    for i in range(N_VIEWS):
        sil, ndc_z, _ = render_sil_and_depth(ctx, verts_gt, faces_gt, mvps[i])
        s = sil[0, :, :, 0].detach().cpu().numpy()
        gt_uint8.append((1.0 - s) * 255)
        gt_depths.append(ndc_z.detach().cpu().numpy())
    gt_uint8  = np.stack(gt_uint8,  axis=0).astype(np.uint8)
    gt_depths = np.stack(gt_depths, axis=0).astype(np.float32)
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                  for i in range(N_VIEWS)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                  for i in range(N_VIEWS)]

    del verts_gt, faces_gt

    # ── initial mesh: 2× Catmull-Clark subdivided icosahedron ────────────
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    _ico = make_icosahedron()
    _ico = catmull_clark(_ico)
    _ico = catmull_clark(_ico)
    _positions, _fcs = mesh_to_arrays(_ico)
    _iv = np.array(_positions, dtype=np.float64)
    _mn, _mx = float(_iv.min()), float(_iv.max())
    _iv = (_iv - (_mn + _mx) / 2.0) * (2.0 / max(_mx - _mn, 1e-6))
    _it = np.array(_fan_triangulate(_fcs), dtype=np.int32)
    verts_np, tris_np = adaptive_remesh(_iv, _it)
    print(f"  Init mesh: V={len(verts_np)} F={len(tris_np)}")

    # ── VLM client ────────────────────────────────────────────────────────
    vlm = None if dry_run else VLMClient()

    # ── optimisation state ────────────────────────────────────────────────
    verts_t, faces_t, opt, sched = _rebuild_opt(
        verts_np, tris_np, cur_step=0, total_steps=total_steps, device=device)
    targets = torch.from_numpy((gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)

    iou_history:   List[float]  = []
    n_injections   = 0
    lap_boost_left = 0
    warmup_left    = 0

    with TrajectoryLogger(out_dir=out_dir, save_obj=False) as traj:

        for step in range(total_steps):
            # ── forward pass ──────────────────────────────────────────────
            opt.zero_grad()
            loss_sil  = torch.tensor(0., device=device)
            loss_depth = torch.tensor(0., device=device)

            w_lap = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
            if lap_boost_left > 0:
                lap_boost_left -= 1

            for i in range(N_VIEWS):
                sil, ndc_z, fg_mask = render_sil_and_depth(
                    ctx, verts_t, faces_t, mvps[i])
                tgt_v = targets[i]   # [H, W, 1]
                loss_sil += F.l1_loss(sil[0], tgt_v)
                loss_depth += depth_loss_masked(ndc_z, fg_mask, gt_depth_t[i], gt_fg_t[i])

            loss_sil   /= N_VIEWS
            loss_depth /= N_VIEWS
            lap = laplacian_loss(verts_t, faces_t)
            edge = edge_length_loss(verts_t, faces_t)

            loss = loss_sil + W_DEPTH * loss_depth + w_lap * lap + W_EDGE * edge
            loss.backward()

            if warmup_left > 0:
                with torch.no_grad():
                    verts_t.grad *= 0.1 * (1.0 - warmup_left / WARMUP_STEPS)
                warmup_left -= 1

            opt.step()
            sched.step()

            # ── periodic eval ─────────────────────────────────────────────
            if step % EVAL_INTERVAL == 0 or step == total_steps - 1:
                with torch.no_grad():
                    pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
                iou = compute_iou_n(pred_sils, gt_uint8)
                iou_history.append(iou)

                if step % 50 == 0:
                    print(f"  step={step:4d}  IoU={iou:.4f}  "
                          f"lap_boost={lap_boost_left}  inj={n_injections}")

            # ── plateau detection → VLM trigger ──────────────────────────
            if (
                step >= MIN_STEP_FOR_INJ
                and n_injections < max_injections
                and _plateau_check(iou_history, PLATEAU_STEPS // EVAL_INTERVAL + 1, PLATEAU_EPS)
            ):
                verts_np = verts_t.detach().cpu().numpy().astype(np.float64)
                tris_np  = faces_t.detach().cpu().numpy()

                # Compute error maps
                error_maps = compute_missing_error_maps(
                    pred_sils, gt_uint8)
                cc_frac = largest_cc_fraction(
                    (pred_sils.mean(0) > 0.5).astype(np.uint8))
                if cc_frac < ERROR_CC_THRESH:
                    print(f"  [VLM] CC too weak ({cc_frac:.3f}), skip injection")
                    iou_history = []   # reset plateau
                    continue

                # Vote on faces
                face_votes, face_dir = vote_faces_multiview_n(
                    error_maps, verts_np, tris_np, mvps)

                # Build candidate regions (error_maps → faithful circle placement)
                candidates = build_candidates(
                    face_votes, face_dir, tris_np, verts_np, mvps,
                    n_candidates=N_CANDIDATES, cluster_size=CLUSTER_SIZE,
                    error_maps=error_maps, img_res=IMG_RES)

                # Encode triptych
                with torch.no_grad():
                    pred_depths_np = render_depths_n(ctx, verts_t, faces_t, mvps)
                png_bytes = encode_state(
                    pred_sils, gt_uint8, pred_depths_np,
                    candidates, step, iou)

                # VLM call
                if dry_run or vlm is None:
                    vlm_reply   = "REGION: 1\nREASON: dry-run auto-pick."
                    region_label = 1
                else:
                    try:
                        vlm_reply    = vlm.ask_with_image(_VLM_SYSTEM_PROMPT, png_bytes)
                        region_label = _parse_vlm_reply(vlm_reply)
                    except Exception as exc:
                        print(f"  [VLM] call failed: {exc}  → picking region 1")
                        vlm_reply    = f"ERROR: {exc}"
                        region_label = 1

                print(f"  [VLM] step={step}  reply={vlm_reply[:80]!r}  "
                      f"→ region {region_label}")

                # Ground region → face cluster
                cluster, extrude_dir = ground_region(
                    region_label, candidates, error_maps,
                    verts_np, tris_np, mvps, face_votes, face_dir)

                # Execute operator
                iou_before = iou
                try:
                    new_verts, new_tris, old_V = execute_operator(
                        "extrude",
                        verts_np=verts_np,
                        tris_np=tris_np,
                        cluster_faces=cluster,
                        extrude_dir=extrude_dir,
                        dist=EXTRUDE_DIST,
                    )
                    print(f"  [EXTRUDE] V {len(verts_np)}→{len(new_verts)}  "
                          f"F {len(tris_np)}→{len(new_tris)}")
                except (AssertionError, RuntimeError, ValueError) as exc:
                    print(f"  [EXTRUDE] FAILED: {exc}  → skip injection")
                    iou_history = []
                    continue

                # Post-injection local edge split
                new_verts, new_tris = split_new_long_edges(new_verts, new_tris, old_V)

                # Log step
                traj.log_step(
                    step          = step,
                    iou_before    = iou_before,
                    candidates    = candidates,
                    vlm_choice    = region_label,
                    vlm_rationale = vlm_reply,
                    op            = "extrude",
                    op_kwargs     = {"dist": EXTRUDE_DIST, "cluster_size": int(len(cluster))},
                    view_png      = png_bytes,
                    verts_np      = new_verts,
                    tris_np       = new_tris,
                )

                # Rebuild Adam on new mesh
                verts_t, faces_t, opt, sched = _rebuild_opt(
                    new_verts, new_tris, cur_step=step,
                    total_steps=total_steps, device=device)
                # targets unchanged — gt_uint8 doesn't change

                lap_boost_left = LAP_WARMUP_STEPS
                warmup_left    = WARMUP_STEPS
                n_injections  += 1
                iou_history    = []   # reset plateau after injection

                # Eval immediately after injection
                with torch.no_grad():
                    pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
                iou_after = compute_iou_n(pred_sils, gt_uint8)
                traj.update_last_iou_after(iou_after)
                print(f"  [POST-INJ] IoU {iou_before:.4f} → {iou_after:.4f} "
                      f"(Δ={iou_after - iou_before:+.4f})")

    # ── final eval ────────────────────────────────────────────────────────
    with torch.no_grad():
        pred_sils_final = render_views_n(ctx, verts_t, faces_t, mvps)
    final_iou = compute_iou_n(pred_sils_final, gt_uint8)
    print(f"\n[VLM LOOP DONE]  final IoU={final_iou:.4f}  injections={n_injections}")

    # Save final render grid
    _save_final_grid(pred_sils_final, gt_uint8, out_path / "final_render.png")

    return {
        "final_iou":       final_iou,
        "n_injections":    n_injections,
        "trajectory_path": str(traj.jsonl_path),
    }


def _save_final_grid(
    pred_sils: np.ndarray,   # [N_V, H, W] float
    gt_uint8:  np.ndarray,   # [N_V, H, W] uint8
    path:      Path,
) -> None:
    """Save a simple 2-column (pred | GT) grid for all views."""
    try:
        from PIL import Image
        import io as _io
        N_V, H, W = pred_sils.shape
        canvas = np.zeros((N_V * H, 2 * W), dtype=np.uint8)
        for vi in range(N_V):
            y = vi * H
            canvas[y:y+H, :W]  = np.clip(pred_sils[vi] * 255, 0, 255).astype(np.uint8)
            canvas[y:y+H, W:]  = ((gt_uint8[vi] < 128).astype(np.uint8) * 255)
        img = Image.fromarray(canvas, "L")
        img.save(str(path))
        print(f"  Saved final render: {path}")
    except Exception as exc:
        print(f"  [WARN] Could not save final render: {exc}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="VLM-driven topology operator loop")
    parser.add_argument("--device",   default="cuda", help="torch device")
    parser.add_argument("--steps",    type=int, default=300,
                        help="Total optimisation steps")
    parser.add_argument("--out_dir",  default="eval_out/vlm_run",
                        help="Output directory for trajectory + renders")
    parser.add_argument("--shape",    default="bunny",
                        help="Shape name: bunny | cow | airplane | torus")
    parser.add_argument("--dry_run",  action="store_true",
                        help="Skip VLM calls (always pick region 1) — for testing")
    parser.add_argument("--max_inj",  type=int, default=MAX_INJECTIONS,
                        help="Max operator injections")
    args = parser.parse_args()

    t0 = time.time()
    result = run_vlm_loop(
        device      = args.device,
        total_steps = args.steps,
        out_dir     = args.out_dir,
        dry_run     = args.dry_run,
        shape       = args.shape,
        max_injections = args.max_inj,
    )
    elapsed = time.time() - t0
    print(f"\nResult: {json.dumps(result, indent=2)}")
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
