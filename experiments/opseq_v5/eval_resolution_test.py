#!/usr/bin/env python3
"""
eval_resolution_test.py — Test the RESOLUTION HYPOTHESIS.

Three independent experiments (extrude-only / stellate-only / menu) all plateaued
at ~0.94 IoU on cow regardless of operator choice, with stellate (which adds
DOF/resolution) the mildest winner. This points to the ceiling being a GLOBAL
resolution / face-count limit rather than an operator-type problem.

This script sweeps the initial mesh resolution (Catmull-Clark subdivision level)
and, optionally, the render resolution, running the multi-trial harness at each
setting. If the IoU ceiling rises decisively (>0.02 above the 2xCC 0.94 baseline)
with more faces / higher render res, the resolution hypothesis is confirmed and
the technical focus should shift from the operator menu to resolution budget.

Usage:
    python3 eval_resolution_test.py --shape cow --cc-levels 2 3 --trials 3
    python3 eval_resolution_test.py --shape cow --cc-levels 2 3 --render-res 256 512 --trials 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

import nvdiffrast.torch as dr
import eval_extrude_v3 as E3
import eval_multitrial as M


def build_init(cc_level: int):
    """Build an init icosphere at the given Catmull-Clark subdivision level."""
    from topmod.primitives  import make_icosahedron
    from topmod.subdivision import catmull_clark
    from topmod.diffgeo     import mesh_to_arrays, _fan_triangulate

    ico = make_icosahedron()
    for _ in range(cc_level):
        ico = catmull_clark(ico)
    positions, fcs = mesh_to_arrays(ico)
    init_verts = np.array(positions, dtype=np.float64)
    mn, mx = float(init_verts.min()), float(init_verts.max())
    init_verts = (init_verts - (mn + mx) / 2.0) * (2.0 / max(mx - mn, 1e-6))
    init_tris  = np.array(_fan_triangulate(fcs), dtype=np.int32)
    return init_verts, init_tris


def build_scene(shape: str, device: str, cc_level: int, render_res: int):
    """Like eval_multitrial.setup_scene but with configurable init CC level and
    render resolution (patches E3.IMG_RES for the duration)."""
    E3.IMG_RES = render_res  # GT + all renders use this
    scene = M.setup_scene(shape, device)   # builds GT at render_res, 2xCC init
    # Override init mesh with requested resolution
    iv, it = build_init(cc_level)
    scene['init_verts'] = iv
    scene['init_tris']  = it
    scene['cc_level']   = cc_level
    scene['render_res'] = render_res
    return scene


def main():
    ap = argparse.ArgumentParser(description="Resolution hypothesis sweep")
    ap.add_argument('--shape',       default='cow')
    ap.add_argument('--cc-levels',   nargs='+', type=int, default=[2, 3])
    ap.add_argument('--render-res',  nargs='+', type=int, default=[256])
    ap.add_argument('--variants',    nargs='+', default=['A', 'C'],
                    choices=['A', 'B', 'C', 'D'])
    ap.add_argument('--trials',      type=int, default=3)
    ap.add_argument('--total-steps', type=int, default=800)
    ap.add_argument('--stellate',    action='store_true',
                    help='force stellate-only for menu variant D (T2=-1)')
    ap.add_argument('--device',      default='cuda')
    ap.add_argument('--json',        default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'
    if args.stellate:
        E3.OP_DEPTH_HF_T2 = -1.0   # force stellate at every plateau for variant D

    inj_flags = dict(
        use_rollback=False, recover_window=E3.RECOVER_WINDOW, use_dist_veto=False,
        use_warm_start=False, use_learnable_dist=False, use_reverse_kick=False,
        use_operator_menu=False,
    )

    print("=" * 78)
    print(f"RESOLUTION SWEEP  shape={args.shape}  cc_levels={args.cc_levels}  "
          f"render_res={args.render_res}  variants={args.variants}  "
          f"trials={args.trials}  steps={args.total_steps}")
    print("Baseline to beat: 2xCC/256px extrude-only C = 0.9434 ± 0.0084")
    print("=" * 78)

    all_results = {}
    for rres in args.render_res:
        for cc in args.cc_levels:
            iv, it = build_init(cc)
            print(f"\n{'#'*70}\n# cc={cc} ({len(it)} tri-faces)  render_res={rres}px\n{'#'*70}")
            scene = build_scene(args.shape, device, cc, rres)
            print(f"  init V={len(scene['init_verts'])} F={len(scene['init_tris'])}  "
                  f"GT rendered at {rres}px")
            for v in args.variants:
                ious, tsum = [], 0.0
                for t in range(args.trials):
                    r = M.run_variant(v, scene, device, args.total_steps, inj_flags)
                    ious.append(r['iou']); tsum += r['time']
                    print(f"    [cc{cc}/{rres}px/{v}] trial {t+1}: "
                          f"IoU={r['iou']:.4f}  n_inj={r['n_inj']}  ({r['time']:.0f}s)")
                s = M.summarize(ious)
                key = f"cc{cc}_{rres}px_{v}"
                all_results[key] = s
                print(f"    → {key}: mean={s['mean']:.4f} ± {s['std']:.4f}  "
                      f"[min {s['min']:.4f} max {s['max']:.4f}]  "
                      f"Δvs0.9434={s['mean']-0.9434:+.4f}")

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    for k, s in all_results.items():
        print(f"  {k:22s}  {s['mean']:.4f} ± {s['std']:.4f}   Δ={s['mean']-0.9434:+.4f}")

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({k: {kk: vv for kk, vv in s.items() if kk != 'values'}
                       | {'values': s['values']} for k, s in all_results.items()},
                      f, indent=2)
        print(f"\nJSON → {args.json}")


if __name__ == '__main__':
    main()
