#!/usr/bin/env python3
"""
eval_multitrial.py — Multi-trial statistical harness for the extrude-injection eval.

WHY THIS EXISTS
---------------
nvdiffrast's antialias BACKWARD pass is nondeterministic: the position-gradient
scatter uses floating-point atomicAdd (antialias.cu AntialiasGradKernel, the
caAtomicAdd3_xyw at ~line 553). Floating-point addition is non-associative, and
the accumulation order depends on GPU block scheduling, so identical inputs give
bit-different gradients run-to-run. This is CONFIRMED by nvdiffrast's author
(NVlabs/nvdiffrast issue #13): "Being fully deterministic would in practice
require removing atomics ... So the answer is no."

Consequence: single-run IoU carries ~1.8% run-to-run variance on our cow/bunny
setup — LARGER than most variant deltas we have been comparing. Seeding does NOT
help (it is atomic ordering, not RNG). The ONLY sound comparison is multi-trial:
run each variant N times, report mean±std, and treat differences smaller than a
noise threshold as indistinguishable.

This driver reuses the exact run functions from eval_extrude_v3 (no logic fork)
and wraps them in an N-trial loop with proper statistics.

Usage:
    python eval_multitrial.py --shape cow --trials 5
    python eval_multitrial.py --shape cow --trials 8 --variants A C
    python eval_multitrial.py --shape bunny --trials 5 --total-steps 800 --json out.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import nvdiffrast.torch as dr

import eval_extrude_v3 as E3
from eval_extrude_v3 import (
    N_VIEWS, IMG_RES, CAMERA_RADIUS, MAX_INJECTIONS, TOTAL_STEPS,
    make_6_cameras, render_sil_and_depth,
    run_baseline, run_with_injection_v3,
    compute_iou_n, compute_depth_l1_n,
)


# ─────────────────────────────────────────────────────────────────────────────
# One-time deterministic setup (forward pass is bit-identical, so GT is stable)
# ─────────────────────────────────────────────────────────────────────────────
def setup_scene(shape: str, device: str):
    """Build ctx, cameras, GT silhouettes/depths, and the init icosphere.

    Everything here is forward-only (bit-identical run-to-run), so it is done
    ONCE and shared across all trials. Only the optimization loops (which invoke
    the nondeterministic antialias backward) vary between trials.
    """
    from eval_real_shapes import load_obj, normalize_to_range, BUNNY_PATH, make_torus

    ctx        = dr.RasterizeCudaContext()
    mvps, eyes = make_6_cameras(radius=CAMERA_RADIUS, device=device)

    _SAMPLE_DIR = os.path.dirname(BUNNY_PATH)
    shape_name  = shape
    if shape == 'torus':
        gt_verts, gt_tris = make_torus()
    elif shape == 'bunny' and os.path.exists(BUNNY_PATH):
        gt_verts, gt_tris = load_obj(BUNNY_PATH)
    else:
        _p = os.path.join(_SAMPLE_DIR, f'{shape}.obj')
        if os.path.exists(_p):
            gt_verts, gt_tris = load_obj(_p)
        elif os.path.exists(BUNNY_PATH):
            print(f"WARNING: {shape} not found — falling back to bunny")
            gt_verts, gt_tris = load_obj(BUNNY_PATH); shape_name = 'bunny'
        else:
            print("WARNING: no shape found — using torus fallback")
            gt_verts, gt_tris = make_torus(); shape_name = 'torus'

    gt_verts = normalize_to_range(gt_verts)

    verts_gt_t = torch.tensor(gt_verts, dtype=torch.float32, device=device)
    faces_gt_t = torch.tensor(gt_tris,  dtype=torch.int32,   device=device)
    gt_sil_views, gt_dep_views = [], []
    with torch.no_grad():
        for i in range(N_VIEWS):
            sil_i, ndc_z_i, _ = render_sil_and_depth(
                ctx, verts_gt_t, faces_gt_t, mvps[i], (IMG_RES, IMG_RES))
            gt_sil_views.append(
                ((1.0 - sil_i[0, :, :, 0].cpu().numpy()) * 255)
                .clip(0, 255).astype(np.uint8))
            gt_dep_views.append(ndc_z_i.cpu().numpy())
    gt_uint8  = np.stack(gt_sil_views,  axis=0)
    gt_depths = np.stack(gt_dep_views,  axis=0)

    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    ico = make_icosahedron()
    ico = catmull_clark(ico); ico = catmull_clark(ico)
    positions, fcs = mesh_to_arrays(ico)
    init_verts = np.array(positions, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    init_verts = (init_verts - (mn + mx) / 2.0) * (2.0 / max(mx - mn, 1e-6))
    init_tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)

    return {
        'ctx': ctx, 'mvps': mvps, 'gt_uint8': gt_uint8, 'gt_depths': gt_depths,
        'init_verts': init_verts, 'init_tris': init_tris, 'shape_name': shape_name,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-trial variant runners (thin wrappers over eval_extrude_v3)
# ─────────────────────────────────────────────────────────────────────────────
def run_variant(variant: str, scene: dict, device: str, total_steps: int,
                inj_flags: dict) -> dict:
    """Run one variant once. Returns {'iou':float, 'depth_l1':float|None,
    'n_inj':int, 'time':float}.

    Variants:
      A — baseline (no injection)
      B — inject, no depth
      C — inject + depth
      D — inject + depth + operator menu (always use_operator_menu=True)
    """
    ctx        = scene['ctx']
    mvps       = scene['mvps']
    gt_uint8   = scene['gt_uint8']
    gt_depths  = scene['gt_depths']
    init_verts = scene['init_verts']
    init_tris  = scene['init_tris']

    t0 = time.time()
    if variant == 'A':
        iou, _ = run_baseline(ctx, init_verts, init_tris, gt_uint8, mvps,
                              device, n_steps=total_steps)
        return {'iou': float(iou), 'depth_l1': None, 'n_inj': 0,
                'time': time.time() - t0}

    use_depth = (variant in ('C', 'D'))
    # Variant D always forces operator menu on
    extra = dict(inj_flags)
    if variant == 'D':
        extra['use_operator_menu'] = True
    iou, _, _, depths, log = run_with_injection_v3(
        ctx, init_verts, init_tris,
        gt_uint8, gt_depths if use_depth else None, mvps, device,
        use_depth=use_depth, max_injections=MAX_INJECTIONS,
        total_steps=total_steps, **extra)
    depth_l1 = float(compute_depth_l1_n(depths, gt_depths, gt_uint8))
    return {'iou': float(iou), 'depth_l1': depth_l1, 'n_inj': len(log),
            'time': time.time() - t0}


# ─────────────────────────────────────────────────────────────────────────────
# Statistics
# ─────────────────────────────────────────────────────────────────────────────
def summarize(values: List[float]) -> dict:
    n = len(values)
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)   # sample std
        std = math.sqrt(var)
    else:
        std = 0.0
    return {'n': n, 'mean': mean, 'std': std,
            'min': min(values), 'max': max(values),
            'spread': max(values) - min(values), 'values': list(values)}


def welch_significant(a: dict, b: dict, noise_floor: float = 0.02) -> Tuple[bool, str]:
    """Is the difference between two variant distributions meaningful?

    Two gates, BOTH must pass to call a difference 'real':
      1. |Δmean| must exceed the absolute noise floor (default 0.02 IoU).
      2. |Δmean| must exceed the combined 1σ of the two distributions
         (a coarse effect-size check; not a formal p-value).
    """
    dmean = b['mean'] - a['mean']
    combined_sigma = math.sqrt(a['std'] ** 2 + b['std'] ** 2)
    gate_floor = abs(dmean) > noise_floor
    gate_sigma = abs(dmean) > combined_sigma
    real = gate_floor and gate_sigma
    if real:
        verdict = "REAL"
    elif abs(dmean) <= noise_floor:
        verdict = f"NOISE (|Δ|={abs(dmean):.4f} ≤ floor {noise_floor})"
    else:
        verdict = (f"INCONCLUSIVE (|Δ|={abs(dmean):.4f} > floor but "
                   f"≤ combined σ {combined_sigma:.4f} — need more trials)")
    return real, verdict


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Multi-trial statistical harness for extrude-injection eval")
    ap.add_argument('--shape',       default='cow', help='bunny | cow | airplane | torus')
    ap.add_argument('--trials',      type=int, default=5, help='trials per variant')
    ap.add_argument('--variants',    nargs='+', default=['A', 'B', 'C'],
                    choices=['A', 'B', 'C', 'D'],
                    help='A=baseline, B=inject no-depth, C=inject+depth, '
                         'D=inject+depth+menu')
    ap.add_argument('--total-steps', type=int, default=TOTAL_STEPS)
    ap.add_argument('--noise-floor', type=float, default=0.02,
                    help='|Δmean| below this is declared NOISE')
    ap.add_argument('--device',      default='cuda')
    ap.add_argument('--json',        default=None, help='write raw results JSON')
    # injection-mode flags (forwarded to run_with_injection_v3)
    ap.add_argument('--rollback',       action='store_true')
    ap.add_argument('--recover-window', type=int, default=E3.RECOVER_WINDOW)
    ap.add_argument('--dist-veto',      action='store_true')
    ap.add_argument('--warm-start',     action='store_true')
    ap.add_argument('--learnable-dist', action='store_true')
    ap.add_argument('--reverse-kick',   action='store_true')
    ap.add_argument('--menu',           action='store_true',
                    help='enable operator menu for variants B and C '
                         '(D always uses menu)')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    inj_flags = dict(
        use_rollback=args.rollback, recover_window=args.recover_window,
        use_dist_veto=args.dist_veto, use_warm_start=args.warm_start,
        use_learnable_dist=args.learnable_dist, use_reverse_kick=args.reverse_kick,
        use_operator_menu=args.menu,
    )

    print("=" * 72)
    print(f"MULTI-TRIAL EVAL  shape={args.shape}  trials={args.trials}  "
          f"variants={args.variants}  steps={args.total_steps}  device={device}")
    print(f"noise_floor={args.noise_floor}   (variance root cause: nvdiffrast "
          f"antialias backward atomicAdd — see nvdiffrast_nondeterminism.md)")
    print("=" * 72)

    scene = setup_scene(args.shape, device)
    print(f"Scene ready: shape={scene['shape_name']}  "
          f"init V={len(scene['init_verts'])} F={len(scene['init_tris'])}")

    results: Dict[str, dict] = {}
    for v in args.variants:
        ious, dls, ninj, tsum = [], [], [], 0.0
        print(f"\n─── Variant {v} × {args.trials} ───")
        for t in range(args.trials):
            r = run_variant(v, scene, device, args.total_steps, inj_flags)
            ious.append(r['iou'])
            if r['depth_l1'] is not None:
                dls.append(r['depth_l1'])
            ninj.append(r['n_inj'])
            tsum += r['time']
            dl_str = f" depth_L1={r['depth_l1']:.4f}" if r['depth_l1'] is not None else ""
            print(f"  trial {t+1}/{args.trials}: IoU={r['iou']:.4f}{dl_str}  "
                  f"n_inj={r['n_inj']}  ({r['time']:.0f}s)")
        s = summarize(ious)
        results[v] = {
            'iou': s,
            'depth_l1': summarize(dls) if dls else None,
            'n_inj_mean': sum(ninj) / len(ninj),
            'total_time': tsum,
        }
        print(f"  → IoU  mean={s['mean']:.4f} ± {s['std']:.4f}  "
              f"[min {s['min']:.4f}, max {s['max']:.4f}, spread {s['spread']:.4f}]")
        if results[v]['depth_l1']:
            ds = results[v]['depth_l1']
            print(f"    depth_L1 mean={ds['mean']:.4f} ± {ds['std']:.4f}")

    # ── Pairwise significance ────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("PAIRWISE COMPARISON (noise floor "
          f"{args.noise_floor}, must also exceed combined σ)")
    print("=" * 72)
    vs = args.variants
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            a, b = results[vs[i]]['iou'], results[vs[j]]['iou']
            dmean = b['mean'] - a['mean']
            _, verdict = welch_significant(a, b, args.noise_floor)
            print(f"  {vs[j]} vs {vs[i]}: Δmean={dmean:+.4f}  →  {verdict}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({'shape': scene['shape_name'], 'trials': args.trials,
                       'total_steps': args.total_steps, 'results': results},
                      f, indent=2)
        print(f"\nRaw results → {args.json}")

    print("\nReminder: differences under the noise floor are NOT evidence — the "
          "antialias backward is nondeterministic and cannot be seeded away.")


if __name__ == '__main__':
    main()
