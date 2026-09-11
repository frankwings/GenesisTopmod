#!/usr/bin/env python3
"""
diag_plateau.py — Diagnose whether vertex displacement can naturally fall
                   below PH_DISP_THRESH (5e-5) given enough steps.

Run a single 2000-step optimisation (no refinement, no face injection) on
armadillo with cc2 icosphere init (480F), identical config to run_phantom_refine.

Outputs:
  eval_out/diag_plateau.json   — milestone numbers
  eval_out/diag_plateau.png    — log-scale displacement + loss curves
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List, Optional

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EVAL_DIR   = os.path.dirname(_SCRIPT_DIR)          # opseq_v5/
_REPO_ROOT  = os.path.dirname(os.path.dirname(_EVAL_DIR))
for _p in (_REPO_ROOT, _EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
import torch.nn.functional as F

import nvdiffrast.torch as dr

from pipeline.cameras            import orbit_cameras
from pipeline.geometry_optimizer import (
    render_silhouette, laplacian_loss, edge_length_loss,
)
from eval_v5          import adaptive_remesh
from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH

from eval_extrude_v3 import (
    make_6_cameras, render_sil_and_depth, render_views_n,
    compute_iou_n, depth_loss_masked,
    dlfl_boundary_stats, _mesh_genus,
    N_VIEWS, IMG_RES, CAMERA_RADIUS,
    W_LAP, W_EDGE, W_DEPTH,
    WARMUP_STEPS, LAP_WARMUP_STEPS, DEPTH_WARMUP_STEPS,
)

from eval_local_refine import (
    LR, LR_MIN,
    PH_EMA_ALPHA, PH_DISP_THRESH, PH_DISP_WINDOW,
)


# ─────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────
TOTAL_STEPS  = 2000
SHAPE        = 'armadillo'
MILESTONES   = [500, 1000, 1500, 2000]
FLOOR_WINDOW = 200     # last N steps to compute "natural floor"


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # ── Scene setup (identical to run_phantom_refine) ───────────────
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    ctx        = dr.RasterizeCudaContext()
    mvps, _    = make_6_cameras(radius=CAMERA_RADIUS, device=device)
    N_v        = mvps.shape[0]

    _SAMPLE_DIR = os.path.dirname(BUNNY_PATH)
    shape_path  = os.path.join(_SAMPLE_DIR, f'{SHAPE}.obj')
    if not os.path.exists(shape_path):
        print(f"ERROR: {shape_path} not found"); return
    gt_verts, gt_tris = load_obj(shape_path)
    gt_verts = normalize_to_range(gt_verts)
    verts_gt_t = torch.tensor(gt_verts, dtype=torch.float32, device=device)
    faces_gt_t = torch.tensor(gt_tris,  dtype=torch.int32,   device=device)

    gt_sil_list, gt_dep_list = [], []
    with torch.no_grad():
        for i in range(N_v):
            sil_i, ndc_z_i, _ = render_sil_and_depth(
                ctx, verts_gt_t, faces_gt_t, mvps[i], (IMG_RES, IMG_RES))
            gt_sil_list.append(
                ((1.0 - sil_i[0, :, :, 0].cpu().numpy()) * 255)
                .clip(0, 255).astype(np.uint8))
            gt_dep_list.append(ndc_z_i.cpu().numpy())
    gt_uint8  = np.stack(gt_sil_list, axis=0)
    gt_depths = np.stack(gt_dep_list, axis=0).astype(np.float32)
    del verts_gt_t, faces_gt_t

    # cc2 init (480F)
    ico = make_icosahedron()
    ico = catmull_clark(ico); ico = catmull_clark(ico)
    positions, fcs = mesh_to_arrays(ico)
    init_verts = np.array(positions, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    init_verts = (init_verts - (mn + mx) / 2.0) * (2.0 / max(mx - mn, 1e-6))
    init_tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)

    verts_np, tris_np = adaptive_remesh(
        init_verts.copy().astype(np.float64), init_tris.copy())
    print(f"Shape={SHAPE}  init V={len(verts_np)} F={len(tris_np)}  "
          f"steps={TOTAL_STEPS}")

    # ── Build tensors + optimiser (same as run_phantom_refine) ──────
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device)
                  for i in range(N_v)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device)
                  for i in range(N_v)]
    targets    = torch.from_numpy(
        (gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)

    verts_t = torch.tensor(verts_np, dtype=torch.float32,
                           device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np, dtype=torch.int32, device=device)
    opt     = torch.optim.Adam([verts_t], lr=LR)
    sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=TOTAL_STEPS, eta_min=LR_MIN)

    # ── Recording arrays ────────────────────────────────────────────
    disp_log:    List[float] = []
    loss_log:    List[float] = []
    ema_log:     List[float] = []
    iou_log:     List[float] = []
    step_log:    List[int]   = []

    loss_ema:    Optional[float] = None
    verts_prev:  Optional[np.ndarray] = None

    t0 = time.time()
    first_below_step: Optional[int] = None

    for step in range(TOTAL_STEPS):
        opt.zero_grad()
        loss_sil   = torch.tensor(0., device=device)
        loss_depth = torch.tensor(0., device=device)

        for i in range(N_v):
            sil, ndc_z, fg = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            loss_sil   = loss_sil + F.l1_loss(sil[0], targets[i])
            loss_depth = loss_depth + depth_loss_masked(
                ndc_z, fg, gt_depth_t[i], gt_fg_t[i])
        loss_sil   = loss_sil / N_v
        loss_depth = loss_depth / N_v
        lap  = laplacian_loss(verts_t, faces_t)
        edge = edge_length_loss(verts_t, faces_t)
        loss = (loss_sil + W_DEPTH * loss_depth + W_LAP * lap + W_EDGE * edge)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step()
        sched.step()

        cur_loss = loss.item()

        # EMA
        if loss_ema is None:
            loss_ema = cur_loss
        else:
            loss_ema = PH_EMA_ALPHA * loss_ema + (1 - PH_EMA_ALPHA) * cur_loss

        # Displacement (identical to run_phantom_refine)
        cur_verts_np = verts_t.detach().cpu().numpy()
        disp = 0.0
        if verts_prev is not None:
            bbox_diag = max(float(cur_verts_np.max() - cur_verts_np.min()), 1e-8)
            disp = float(np.mean(np.linalg.norm(
                cur_verts_np - verts_prev, axis=1))) / bbox_diag
        verts_prev = cur_verts_np.copy()

        # Record
        disp_log.append(disp)
        loss_log.append(cur_loss)
        ema_log.append(loss_ema)
        step_log.append(step)

        # First time below threshold?
        if first_below_step is None and disp < PH_DISP_THRESH and step > 0:
            first_below_step = step

        # IoU every 50 steps
        if step % 50 == 0:
            with torch.no_grad():
                pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
            iou = compute_iou_n(pred_sils, gt_uint8)
            iou_log.append((step, iou))

        # Print every 200 steps
        if step % 200 == 0:
            iou_str = f"  IoU={iou_log[-1][1]:.4f}" if iou_log else ""
            print(f"  step={step:5d}  loss={cur_loss:.5f}  ema={loss_ema:.5f}"
                  f"  disp={disp:.2e}{iou_str}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # ── Compute summaries ───────────────────────────────────────────
    disp_arr = np.array(disp_log)

    # Milestones
    milestones = {}
    for ms in MILESTONES:
        idx = min(ms, len(disp_arr)) - 1
        if idx >= 0:
            # Window average at milestone
            win_start = max(0, idx - PH_DISP_WINDOW + 1)
            milestones[str(ms)] = {
                'disp_instant': float(disp_arr[idx]),
                'disp_window_avg': float(np.mean(disp_arr[win_start:idx+1])),
            }

    # Natural floor: mean of last FLOOR_WINDOW steps
    floor_start = max(0, len(disp_arr) - FLOOR_WINDOW)
    natural_floor = float(np.mean(disp_arr[floor_start:]))
    natural_floor_std = float(np.std(disp_arr[floor_start:]))

    # Loss floor
    loss_arr = np.array(loss_log)
    loss_floor = float(np.mean(loss_arr[floor_start:]))

    result = {
        'shape':             SHAPE,
        'total_steps':       TOTAL_STEPS,
        'init_F':            int(len(tris_np)),
        'PH_DISP_THRESH':   PH_DISP_THRESH,
        'LR':                LR,
        'LR_MIN':            LR_MIN,
        'first_below_5e-5':  first_below_step,
        'natural_floor_disp': natural_floor,
        'natural_floor_std':  natural_floor_std,
        'loss_floor':         loss_floor,
        'milestones':         milestones,
        'final_iou':          float(iou_log[-1][1]) if iou_log else 0.0,
        'elapsed_s':          elapsed,
        'threshold_reachable': first_below_step is not None,
        'ratio_floor_to_thresh': natural_floor / PH_DISP_THRESH,
    }

    # ── Save JSON ───────────────────────────────────────────────────
    json_path = os.path.join(out_dir, 'diag_plateau.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\nJSON → {json_path}")

    # ── Summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PLATEAU DIAGNOSTIC SUMMARY")
    print("=" * 60)
    print(f"  Shape:               {SHAPE}")
    print(f"  Init faces:          {len(tris_np)}")
    print(f"  Total steps:         {TOTAL_STEPS}")
    print(f"  LR range:            {LR} → {LR_MIN}")
    print(f"  PH_DISP_THRESH:      {PH_DISP_THRESH:.1e}")
    print(f"  First below 5e-5:    {first_below_step or 'NEVER'}")
    print(f"  Natural floor (last {FLOOR_WINDOW}): "
          f"{natural_floor:.2e} ± {natural_floor_std:.2e}")
    print(f"  Floor / threshold:   {natural_floor / PH_DISP_THRESH:.1f}×")
    print(f"  Loss floor:          {loss_floor:.5f}")
    print(f"  Final IoU:           {result['final_iou']:.4f}")
    for ms_str, ms_data in milestones.items():
        print(f"  disp @ step {ms_str:>5}: "
              f"instant={ms_data['disp_instant']:.2e}  "
              f"win_avg={ms_data['disp_window_avg']:.2e}")
    if first_below_step is not None:
        print(f"\n  ✓ Threshold IS reachable — need ≥{first_below_step} steps")
        print(f"    Root cause: cold-start + insufficient inter-refine steps")
        print(f"    Fix: warm-start optimizer + give ≥{first_below_step} steps/round")
    else:
        print(f"\n  ✗ Threshold NOT reachable in {TOTAL_STEPS} steps")
        print(f"    Floor is {natural_floor/PH_DISP_THRESH:.1f}× above threshold")
        print(f"    Fix: use relative threshold (e.g., 10% of peak displacement)")
    print("=" * 60)

    # ── Plot ────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        steps = np.arange(len(disp_arr))

        # ── Top panel: displacement ──
        ax1.semilogy(steps, disp_arr, color='#2196F3', alpha=0.3,
                     linewidth=0.5, label='instant disp')
        # Rolling window average
        if len(disp_arr) >= PH_DISP_WINDOW:
            rolling = np.convolve(disp_arr,
                                  np.ones(PH_DISP_WINDOW) / PH_DISP_WINDOW,
                                  mode='valid')
            ax1.semilogy(np.arange(PH_DISP_WINDOW - 1, len(disp_arr)),
                         rolling, color='#1565C0', linewidth=1.5,
                         label=f'{PH_DISP_WINDOW}-step window avg')
        ax1.axhline(y=PH_DISP_THRESH, color='red', linestyle='--',
                    linewidth=1.5, label=f'PH_DISP_THRESH = {PH_DISP_THRESH:.0e}')
        ax1.axhline(y=natural_floor, color='green', linestyle=':',
                    linewidth=1.5,
                    label=f'natural floor = {natural_floor:.2e}')
        if first_below_step is not None:
            ax1.axvline(x=first_below_step, color='orange', linestyle='-.',
                        linewidth=1, label=f'first below @ step {first_below_step}')
        ax1.set_ylabel('Vertex displacement (norm.)')
        ax1.set_title(f'Plateau Diagnostic: {SHAPE} cc2→480F, '
                      f'{TOTAL_STEPS} steps, LR={LR}→{LR_MIN}')
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Bottom panel: loss + IoU ──
        ax2.semilogy(steps, loss_arr, color='#FF9800', alpha=0.3,
                     linewidth=0.5, label='loss (instant)')
        ax2.semilogy(steps, ema_log, color='#E65100', linewidth=1.5,
                     label=f'loss EMA (α={PH_EMA_ALPHA})')
        ax2.set_ylabel('Loss', color='#E65100')
        ax2.set_xlabel('Step')
        ax2.grid(True, alpha=0.3)

        ax2b = ax2.twinx()
        iou_steps = [s for s, _ in iou_log]
        iou_vals  = [v for _, v in iou_log]
        ax2b.plot(iou_steps, iou_vals, color='#4CAF50', linewidth=1.5,
                  marker='.', markersize=3, label='IoU')
        ax2b.set_ylabel('IoU', color='#4CAF50')
        ax2b.set_ylim(0, 1)

        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2,
                   loc='center right', fontsize=8)

        plt.tight_layout()
        png_path = os.path.join(out_dir, 'diag_plateau.png')
        plt.savefig(png_path, dpi=150)
        plt.close()
        print(f"PNG → {png_path}")
    except ImportError:
        print("matplotlib not available — skipping plot")


if __name__ == '__main__':
    main()
