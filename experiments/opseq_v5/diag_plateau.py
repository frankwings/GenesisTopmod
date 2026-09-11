#!/usr/bin/env python3
"""
diag_plateau.py — Diagnose plateau trigger (condition B) for phantom-apex refine.

Runs 2000 steps on armadillo (cc2 480F) with NO face additions using the
EXACT same loss + LR + warmup configuration as run_phantom_refine in
eval_local_refine.py.  Records per-step vertex displacement, loss EMA, and IoU.

Outputs
-------
  eval_out/diag_plateau.png  — disp + loss decay curves (log-y)
  eval_out/diag_plateau.json — summary statistics

Key question
------------
Does the 50-step windowed displacement naturally drop below PH_DISP_THRESH=5e-5?
If yes → root cause is insufficient steps / cold-start churn; fix = warm restart.
If no  → threshold is unreachable at this LR; fix = relative auto-calibration.

Usage
-----
  python3 experiments/opseq_v5/diag_plateau.py
  python3 experiments/opseq_v5/diag_plateau.py --steps 2000 --shape armadillo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import nvdiffrast.torch as dr

from pipeline.geometry_optimizer import laplacian_loss, edge_length_loss
from eval_v5          import adaptive_remesh
from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH, make_torus

from eval_extrude_v3 import (
    make_6_cameras, render_sil_and_depth, render_views_n,
    compute_iou_n, depth_loss_masked,
    N_VIEWS, IMG_RES, CAMERA_RADIUS,
    W_LAP, W_LAP_BOOST, W_EDGE, W_DEPTH,
    WARMUP_STEPS, LAP_WARMUP_STEPS, DEPTH_WARMUP_STEPS,
)

# ── Mirror constants from eval_local_refine.py / run_phantom_refine ───────────
LR             = 3e-3
LR_MIN         = 3e-5
PH_EMA_ALPHA   = 0.95
PH_DISP_THRESH = 5e-5     # condition-B threshold under investigation
PH_DISP_WINDOW = 50       # sliding window for displacement average

# Armadillo path (in pymeshlab samples)
ARMADILLO_PATH = (
    "/home/kingy/Projects/Genesis/GenesisExp/GenesisHunyuan/"
    ".venv/lib/python3.12/site-packages/pymeshlab/tests/sample_meshes/armadillo.obj"
)

EVAL_INTERVAL = 20   # render IoU every N steps (expensive, keep sparse)
DISP_REPORT_STEPS = [500, 1000, 1500, 2000]


def run_diag(shape: str, total_steps: int, device: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # ── Scene setup ──────────────────────────────────────────────────────────
    ctx        = dr.RasterizeCudaContext()
    mvps, _    = make_6_cameras(radius=CAMERA_RADIUS, device=device)

    _SAMPLE_DIR = os.path.dirname(BUNNY_PATH)

    if shape == 'armadillo' and os.path.exists(ARMADILLO_PATH):
        gt_verts, gt_tris = load_obj(ARMADILLO_PATH)
        shape_used = 'armadillo'
    elif shape == 'bunny' and os.path.exists(BUNNY_PATH):
        gt_verts, gt_tris = load_obj(BUNNY_PATH)
        shape_used = 'bunny'
    else:
        _p = os.path.join(_SAMPLE_DIR, f'{shape}.obj')
        if os.path.exists(_p):
            gt_verts, gt_tris = load_obj(_p)
            shape_used = shape
        elif os.path.exists(BUNNY_PATH):
            print(f"  WARNING: {shape!r} not found → falling back to bunny")
            gt_verts, gt_tris = load_obj(BUNNY_PATH)
            shape_used = 'bunny'
        else:
            gt_verts, gt_tris = make_torus()
            shape_used = 'torus'

    print(f"Shape: {shape_used}  GT V={len(gt_verts)} F={len(gt_tris)}")

    gt_verts = normalize_to_range(gt_verts)
    vt_gt = torch.tensor(gt_verts, dtype=torch.float32, device=device)
    ft_gt = torch.tensor(gt_tris,  dtype=torch.int32,   device=device)

    gt_sils, gt_deps = [], []
    with torch.no_grad():
        for i in range(N_VIEWS):
            sil_i, ndc_z_i, _ = render_sil_and_depth(
                ctx, vt_gt, ft_gt, mvps[i], (IMG_RES, IMG_RES))
            gt_sils.append(
                ((1.0 - sil_i[0, :, :, 0].cpu().numpy()) * 255)
                .clip(0, 255).astype(np.uint8))
            gt_deps.append(ndc_z_i.cpu().numpy())
    del vt_gt, ft_gt
    gt_uint8  = np.stack(gt_sils, axis=0)
    gt_depths = np.stack(gt_deps, axis=0).astype(np.float32)

    # ── cc2 icosphere init (~480F) ────────────────────────────────────────────
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    ico = make_icosahedron()
    ico = catmull_clark(ico); ico = catmull_clark(ico)
    pos, fcs = mesh_to_arrays(ico)
    init_verts = np.array(pos, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    init_verts = (init_verts - (mn + mx) / 2.0) * (2.0 / max(mx - mn, 1e-6))
    init_tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)

    verts_np, tris_np = adaptive_remesh(
        init_verts.copy().astype(np.float64), init_tris.copy())
    print(f"Init mesh: V={len(verts_np)} F={len(tris_np)}")

    verts_t = torch.tensor(verts_np, dtype=torch.float32, device=device).requires_grad_(True)
    faces_t = torch.tensor(tris_np,  dtype=torch.int32,   device=device)

    targets    = torch.from_numpy((gt_uint8 < 128).astype(np.float32)).unsqueeze(-1).to(device)
    gt_depth_t = [torch.from_numpy(gt_depths[i]).float().to(device) for i in range(N_VIEWS)]
    gt_fg_t    = [torch.from_numpy(gt_uint8[i] < 128).to(device) for i in range(N_VIEWS)]

    opt   = torch.optim.Adam([verts_t], lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps, eta_min=LR_MIN)

    # ── Per-step tracking ─────────────────────────────────────────────────────
    disp_history: List[float] = []   # per-step raw displacement (normalised)
    disp_window:  List[float] = []   # 50-step windowed avg at every step
    loss_history: List[float] = []
    ema_history:  List[float] = []
    iou_history:  List[float] = []   # sparse (every EVAL_INTERVAL steps)
    iou_steps:    List[int]   = []

    loss_ema  = None
    verts_prev = None
    warmup_counter   = WARMUP_STEPS
    lap_boost_left   = LAP_WARMUP_STEPS
    depth_boost_left = DEPTH_WARMUP_STEPS

    t0 = time.time()
    for step in range(total_steps):

        # ── Warmup LR (mirrors run_phantom_refine exactly) ───────────────────
        if warmup_counter > 0:
            frac = 1.0 - warmup_counter / WARMUP_STEPS
            for pg in opt.param_groups:
                pg['lr'] = LR * max(frac, 0.05)
            warmup_counter -= 1

        # ── Forward ──────────────────────────────────────────────────────────
        opt.zero_grad()
        l_sil = torch.tensor(0., device=device)
        l_dep = torch.tensor(0., device=device)
        w_lap = W_LAP_BOOST if lap_boost_left > 0 else W_LAP
        w_dm  = ((DEPTH_WARMUP_STEPS - depth_boost_left) / DEPTH_WARMUP_STEPS
                 if depth_boost_left > 0 else 1.0)
        if lap_boost_left   > 0: lap_boost_left   -= 1
        if depth_boost_left > 0: depth_boost_left -= 1

        for i in range(N_VIEWS):
            sil, ndc_z, fg_mask = render_sil_and_depth(
                ctx, verts_t, faces_t, mvps[i], (IMG_RES, IMG_RES))
            l_sil += F.l1_loss(sil[0], targets[i])
            l_dep += depth_loss_masked(ndc_z, fg_mask, gt_depth_t[i], gt_fg_t[i])
        l_sil /= N_VIEWS; l_dep /= N_VIEWS
        loss = (l_sil + W_DEPTH * w_dm * l_dep
                + w_lap * laplacian_loss(verts_t, faces_t)
                + W_EDGE * edge_length_loss(verts_t, faces_t))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([verts_t], 1.0)
        opt.step(); sched.step()

        cur_loss = loss.item()
        loss_ema = cur_loss if loss_ema is None \
            else PH_EMA_ALPHA * loss_ema + (1 - PH_EMA_ALPHA) * cur_loss

        # ── Displacement (exact same formula as eval_local_refine.py L802-804) ─
        cur_verts_np = verts_t.detach().cpu().numpy()
        if verts_prev is not None and cur_verts_np.shape == verts_prev.shape:
            bbox_diag = max(float(cur_verts_np.max() - cur_verts_np.min()), 1e-8)
            raw_disp  = float(np.mean(np.linalg.norm(
                cur_verts_np - verts_prev, axis=1))) / bbox_diag
            disp_history.append(raw_disp)
        else:
            raw_disp = float('nan')
        verts_prev = cur_verts_np.copy()

        # 50-step windowed average (same as trigger condition B)
        if len(disp_history) >= PH_DISP_WINDOW:
            win_avg = float(np.mean(disp_history[-PH_DISP_WINDOW:]))
        elif len(disp_history) > 0:
            win_avg = float(np.mean(disp_history))
        else:
            win_avg = float('nan')
        disp_window.append(win_avg)

        loss_history.append(cur_loss)
        ema_history.append(loss_ema)

        # ── Sparse IoU eval ───────────────────────────────────────────────────
        if step % EVAL_INTERVAL == 0 or step == total_steps - 1:
            with torch.no_grad():
                pred_sils = render_views_n(ctx, verts_t, faces_t, mvps)
            iou = compute_iou_n(pred_sils, gt_uint8)
            iou_history.append(iou)
            iou_steps.append(step)
            if step % 200 == 0:
                print(f"  step={step:4d}  loss={cur_loss:.5f}  ema={loss_ema:.5f}"
                      f"  disp_raw={raw_disp:.2e}  disp_win={win_avg:.2e}"
                      f"  IoU={iou:.4f}"
                      f"  ({time.time()-t0:.0f}s)")

    total_time = time.time() - t0
    print(f"\n  Finished {total_steps} steps in {total_time:.1f}s")

    # ── Analysis ──────────────────────────────────────────────────────────────
    disp_win_arr = np.array([v for v in disp_window if not np.isnan(v)])
    steps_with_data = np.arange(1, len(disp_window) + 1)  # step 0 has no prev

    # First step where windowed avg drops below threshold
    below_thresh = np.where(disp_win_arr < PH_DISP_THRESH)[0]
    first_below_step = int(steps_with_data[below_thresh[0]]) if len(below_thresh) > 0 else None

    # Values at report steps (use window avg)
    def _win_at(s):
        idx = min(s, len(disp_window) - 1)
        v = disp_window[idx]
        return float(v) if not np.isnan(v) else None

    natural_floor_last200 = (float(np.mean(disp_win_arr[-200:]))
                              if len(disp_win_arr) >= 200 else float(np.mean(disp_win_arr)))
    loss_floor_last200 = float(np.mean(loss_history[-200:])) if loss_history else 0.

    # Step where loss basically stops (loss decreases < 1% per 100 steps)
    loss_arr = np.array(loss_history)
    loss_converge_step = None
    for s in range(100, len(loss_arr), 100):
        drop_pct = (loss_arr[s - 100] - loss_arr[s]) / max(loss_arr[s - 100], 1e-8)
        if drop_pct < 0.01:
            loss_converge_step = s
            break

    # ── JSON output ───────────────────────────────────────────────────────────
    result = {
        'shape':         shape_used,
        'total_steps':   total_steps,
        'ph_disp_thresh': PH_DISP_THRESH,
        'ph_disp_window': PH_DISP_WINDOW,
        'first_below_thresh_step': first_below_step,
        'disp_win_at_steps': {str(s): _win_at(s) for s in DISP_REPORT_STEPS},
        'natural_floor_last200':  natural_floor_last200,
        'loss_floor_last200':     loss_floor_last200,
        'loss_approx_converge_step': loss_converge_step,
        'total_time_s': total_time,
        'verdict': (
            f"disp drops below {PH_DISP_THRESH:.0e} at step {first_below_step}"
            if first_below_step is not None
            else f"disp NEVER drops below {PH_DISP_THRESH:.0e} in {total_steps} steps"
        ),
        'recommendation': (
            "ROOT CAUSE: insufficient steps + cold-start churn. FIX: warm restart + allow more steps per round."
            if first_below_step is not None and first_below_step < 600
            else (
                "ROOT CAUSE: threshold unreachable at LR=3e-3. FIX: relative auto-calibration (e.g. 10% of peak)."
                if first_below_step is None
                else f"ROOT CAUSE: long convergence (step {first_below_step}). FIX: more steps or relative threshold."
            )
        ),
    }

    json_path = os.path.join(out_dir, 'diag_plateau.json')
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"\n  JSON → {json_path}")

    # ── PNG output ────────────────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        steps_all = np.arange(len(loss_history))

        # Panel 1: displacement (log-y)
        ax = axes[0]
        # Raw per-step disp (skip step 0 which has no prev)
        disp_raw_arr = np.array([v for v in disp_history])
        ax.semilogy(np.arange(1, len(disp_raw_arr) + 1), disp_raw_arr,
                    alpha=0.25, color='steelblue', lw=0.8, label='raw disp/step')
        if len(disp_win_arr) > 0:
            ax.semilogy(steps_with_data, disp_win_arr,
                        color='steelblue', lw=1.8, label=f'{PH_DISP_WINDOW}-step avg')
        ax.axhline(PH_DISP_THRESH, color='red', ls='--', lw=1.5,
                   label=f'PH_DISP_THRESH={PH_DISP_THRESH:.0e}')
        if first_below_step is not None:
            ax.axvline(first_below_step, color='green', ls=':', lw=1.5,
                       label=f'first below threshold (step {first_below_step})')
        ax.set_ylabel('Norm. vertex displacement', fontsize=10)
        ax.set_title(f'Plateau diagnostic — {shape_used}  (cc2 480F, {total_steps} steps, LR={LR})',
                     fontsize=11)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, which='both', alpha=0.3)

        # Panel 2: loss + EMA (log-y)
        ax = axes[1]
        ax.semilogy(steps_all, loss_history, alpha=0.3, color='orange', lw=0.8, label='loss')
        ax.semilogy(steps_all, ema_history,  color='darkorange', lw=1.8, label='loss EMA')
        if loss_converge_step is not None:
            ax.axvline(loss_converge_step, color='gray', ls=':', lw=1.5,
                       label=f'approx converge (step {loss_converge_step})')
        ax.set_ylabel('Loss', fontsize=10)
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, which='both', alpha=0.3)

        # Panel 3: IoU
        ax = axes[2]
        ax.plot(iou_steps, iou_history, color='green', lw=1.5, label='IoU')
        ax.set_xlabel('Step', fontsize=10)
        ax.set_ylabel('IoU', fontsize=10)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

        # Annotate floor
        ax2 = axes[0]
        if disp_win_arr.size > 0:
            ax2.annotate(f'floor={natural_floor_last200:.2e}',
                         xy=(total_steps * 0.85, natural_floor_last200 * 1.5),
                         fontsize=8, color='steelblue')

        plt.tight_layout()
        png_path = os.path.join(out_dir, 'diag_plateau.png')
        plt.savefig(png_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  PNG  → {png_path}")
    except Exception as e:
        print(f"  [WARN] matplotlib failed: {e}")
        png_path = None

    # ── Console report ────────────────────────────────────────────────────────
    print("\n" + "=" * 62)
    print("DIAGNOSIS REPORT")
    print("=" * 62)
    print(f"  Shape:       {shape_used}")
    print(f"  Steps:       {total_steps}")
    print(f"  Threshold:   PH_DISP_THRESH = {PH_DISP_THRESH:.0e}")
    print(f"  Window:      PH_DISP_WINDOW = {PH_DISP_WINDOW} steps")
    print()
    print(f"  Windowed displacement at key steps:")
    for s in DISP_REPORT_STEPS:
        v = _win_at(s)
        vs = f"{v:.3e}" if v is not None else "N/A"
        marker = " ← BELOW THRESH" if (v is not None and v < PH_DISP_THRESH) else ""
        print(f"    step {s:4d}: {vs}{marker}")
    print()
    print(f"  First step below threshold: {first_below_step}")
    print(f"  Natural floor (last 200 steps avg): {natural_floor_last200:.3e}")
    print(f"  Loss approx. converge step: {loss_converge_step}")
    print(f"  Loss floor (last 200 steps):  {loss_floor_last200:.5f}")
    print()
    print(f"  VERDICT: {result['verdict']}")
    print(f"  REC:     {result['recommendation']}")
    print("=" * 62)

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shape',  default='armadillo',
                    help='Shape to use (armadillo, cow, bunny, torus)')
    ap.add_argument('--steps',  type=int, default=2000)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out',    default=None,
                    help='Output directory (default: eval_out/ next to this script)')
    args = ap.parse_args()

    device  = args.device if torch.cuda.is_available() else 'cpu'
    out_dir = args.out or os.path.join(_SCRIPT_DIR, 'eval_out')

    run_diag(
        shape       = args.shape,
        total_steps = args.steps,
        device      = device,
        out_dir     = out_dir,
    )


if __name__ == '__main__':
    main()
