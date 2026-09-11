#!/usr/bin/env python3
"""
eval_liou_h2h.py — Head-to-head: Local-IoU-deficit (LIOU) vs random refinement.

Experiment
----------
  Methods : LIOU (local IoU deficit + depth, worst-view) / LIOU_RND (random)
  Budget  : 800F (tight, from cc2=480F start)
  Shape   : armadillo (primary)
  Trials  : 4 (Welch t-test, noise_floor=0.02)

Validity gate
-------------
  Each round logs trigger type ('stall' or 'force').  A trial with ANY
  force-triggered round is INVALID and excluded from the H1 test.
  This ensures we only compare runs where the loss-EMA stall mechanism
  genuinely fired (principled trigger, not safety valve).

H1
--
  LIOU significantly > LIOU_RND at tight budget (800F).
  If H1 fails, the LIOU approach is rejected.

Usage
-----
  python3 eval_liou_h2h.py --shapes armadillo --budgets 800 --trials 4

  # Via bg_task for long runs:
  python3 -m framework.bg_task launch --id liou_h1_armadillo --timeout 7200 \\
      --cwd /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5 -- \\
      python3 -u eval_liou_h2h.py --shapes armadillo --budgets 800 --trials 4 \\
      --json eval_out/liou_h1_armadillo.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
for _p in (_REPO_ROOT, _SCRIPT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import torch

from eval_local_refine import (
    setup_scene, run_variant, summarize,
    LIOU_K,
)


def welch_significant(a: dict, b: dict,
                      noise_floor: float = 0.02) -> Tuple[bool, str]:
    """Two-gate significance: |dmean| > floor AND |dmean| > combined sigma."""
    dmean = b['mean'] - a['mean']
    combined_sigma = math.sqrt(a['std'] ** 2 + b['std'] ** 2)
    gate_floor = abs(dmean) > noise_floor
    gate_sigma = abs(dmean) > combined_sigma
    real = gate_floor and gate_sigma
    if real:
        verdict = "REAL"
    elif abs(dmean) <= noise_floor:
        verdict = f"NOISE (|d|={abs(dmean):.4f} <= floor {noise_floor})"
    else:
        verdict = (f"INCONCLUSIVE (|d|={abs(dmean):.4f} > floor but "
                   f"<= combined sigma {combined_sigma:.4f})")
    return real, verdict


def run_one_cell(
    scene: dict, device: str, variant: str,
    budget: int, total_steps: int, trials: int,
) -> dict:
    """Run `trials` repetitions of a variant at a given face budget.

    Returns summary dict with iou stats, validity info, and per-trial logs.
    """
    ious, faces_list, rounds_list, times = [], [], [], []
    valid_ious = []       # only from trials where trial_valid=True
    n_valid    = 0
    n_invalid  = 0
    per_trial  = []

    for t in range(trials):
        r = run_variant(variant, scene, device, total_steps,
                        face_budget=budget, liou_K=LIOU_K)
        ious.append(r['iou'])
        faces_list.append(r['final_faces'])
        rounds_list.append(r.get('n_rounds', 0))
        times.append(r.get('time', 0))

        trial_valid = r.get('trial_valid', True)  # non-LIOU variants default valid
        if trial_valid:
            valid_ious.append(r['iou'])
            n_valid += 1
        else:
            n_invalid += 1

        # Summarize trigger types from refine log
        log = r.get('log', [])
        triggers = [entry.get('trigger_type', '?') for entry in log]
        trigger_summary = ', '.join(triggers) if triggers else 'no-rounds'

        print(f"    trial {t+1}/{trials}: IoU={r['iou']:.4f}  "
              f"F={r['final_faces']}  rounds={r.get('n_rounds',0)}  "
              f"extrudes={r.get('n_extrudes',0)}  "
              f"valid={trial_valid}  triggers=[{trigger_summary}]  "
              f"({r.get('time',0):.0f}s)")

        per_trial.append({
            'iou':        r['iou'],
            'faces':      r['final_faces'],
            'n_rounds':   r.get('n_rounds', 0),
            'n_extrudes': r.get('n_extrudes', 0),
            'valid':      trial_valid,
            'triggers':   triggers,
            'time':       r.get('time', 0),
        })

    s_all   = summarize(ious)
    s_valid = summarize(valid_ious)

    return {
        'iou':          s_all,
        'iou_valid':    s_valid,
        'faces_mean':   sum(faces_list) / len(faces_list),
        'rounds_mean':  sum(rounds_list) / len(rounds_list),
        'time_mean':    sum(times) / len(times),
        'n_valid':      n_valid,
        'n_invalid':    n_invalid,
        'per_trial':    per_trial,
    }


def main():
    ap = argparse.ArgumentParser(
        description="LIOU (Local-IoU-deficit) H2H experiment")
    ap.add_argument('--shapes',      nargs='+', default=['armadillo'])
    ap.add_argument('--budgets',     nargs='+', type=int, default=[800])
    ap.add_argument('--variants',    nargs='+', default=['LIOU', 'LIOU_RND'],
                    choices=['LIOU', 'LIOU_RND', 'LIOU_EX'])
    ap.add_argument('--trials',      type=int, default=4)
    ap.add_argument('--total-steps', type=int, default=2400,
                    help='Total optimisation steps (should be high enough '
                         'for natural stall convergence)')
    ap.add_argument('--device',      default='cuda')
    ap.add_argument('--json',        default=None)
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    # Baseline references (from liou_h1_cow.json, n=6 trials)
    BASELINE_LIOU     = {'mean': 0.9459, 'std': 0.0041, 'n': 6}
    BASELINE_LIOU_RND = {'mean': 0.9494, 'std': 0.0053, 'n': 6}
    BASELINE_CC3      = {'mean': 0.9662, 'std': 0.0007, 'n': 6}

    print("=" * 72)
    print(f"LIOU H2H EXPERIMENT")
    print(f"  shapes   = {args.shapes}")
    print(f"  budgets  = {args.budgets}")
    print(f"  variants = {args.variants}")
    print(f"  trials   = {args.trials}")
    print(f"  steps    = {args.total_steps}")
    print(f"  K        = {LIOU_K}")
    print(f"  Validity gate: 'stall' or 'extrude' trigger counts as valid")
    print(f"  Baselines @800F,cow: LIOU={BASELINE_LIOU['mean']:.4f}±{BASELINE_LIOU['std']:.4f}  "
          f"LIOU_RND={BASELINE_LIOU_RND['mean']:.4f}±{BASELINE_LIOU_RND['std']:.4f}  "
          f"cc3={BASELINE_CC3['mean']:.4f}±{BASELINE_CC3['std']:.4f}")
    print("=" * 72)

    all_results: Dict[str, dict] = {}

    for shape in args.shapes:
        print(f"\n{'=' * 72}")
        print(f"SHAPE: {shape}")
        print(f"{'=' * 72}")

        scene = setup_scene(shape, device)
        actual_shape = scene['shape_name']
        if actual_shape != shape:
            print(f"  WARNING: {shape} not found, using {actual_shape}")

        shape_results: Dict[str, Dict[str, dict]] = {}

        for budget in args.budgets:
            print(f"\n  == Budget {budget}F ==")
            budget_results: Dict[str, dict] = {}

            for v in args.variants:
                print(f"\n  --- {v} x {args.trials} @ {budget}F ---")
                cell = run_one_cell(
                    scene, device, v, budget, args.total_steps, args.trials)
                budget_results[v] = cell

                s = cell['iou']
                sv = cell['iou_valid']
                print(f"  -> {v} (all):   IoU={s['mean']:.4f} +/- {s['std']:.4f}  "
                      f"[{s['min']:.4f}, {s['max']:.4f}]  "
                      f"F_mean={cell['faces_mean']:.0f}  "
                      f"rounds={cell['rounds_mean']:.1f}  "
                      f"time={cell['time_mean']:.0f}s")
                print(f"  -> {v} (valid): IoU={sv['mean']:.4f} +/- {sv['std']:.4f}  "
                      f"n_valid={cell['n_valid']}/{args.trials}  "
                      f"n_invalid={cell['n_invalid']}")

            shape_results[str(budget)] = budget_results

            # ── H1 test using VALID trials only ──
            print(f"\n  H1 TEST @ {budget}F (validity-gated, noise_floor=0.02):")
            vs = args.variants
            if len(vs) >= 2:
                a_cell = budget_results[vs[0]]
                b_cell = budget_results[vs[1]]
                a_valid = a_cell['iou_valid']
                b_valid = b_cell['iou_valid']

                if a_valid['n'] < 2 or b_valid['n'] < 2:
                    print(f"    INSUFFICIENT VALID TRIALS: "
                          f"{vs[0]}={a_valid['n']}, {vs[1]}={b_valid['n']}  "
                          f"(need >= 2 each)")
                    print(f"    H1 INCONCLUSIVE (not enough valid trials)")
                else:
                    real, verdict = welch_significant(b_valid, a_valid)
                    dmean = a_valid['mean'] - b_valid['mean']
                    h1_pass = real and dmean > 0
                    print(f"    {vs[0]} (valid): {a_valid['mean']:.4f} +/- {a_valid['std']:.4f}  "
                          f"(n={a_valid['n']})")
                    print(f"    {vs[1]} (valid): {b_valid['mean']:.4f} +/- {b_valid['std']:.4f}  "
                          f"(n={b_valid['n']})")
                    print(f"    dmean = {dmean:+.4f}  -> {verdict}")
                    print(f"    H1 {'PASS' if h1_pass else 'FAIL'}")
                    if not h1_pass:
                        print(f"    WARNING: H1 FAILED -- LIOU does NOT significantly "
                              f"beat random at {budget}F. Approach rejected.")

            # ── Compare each variant against stored LIOU_RND baseline ──
            for v in vs:
                if v in budget_results:
                    cell_v = budget_results[v]['iou_valid']
                    if cell_v['n'] < 2:
                        print(f"    {v} vs baseline LIOU_RND: "
                              f"INSUFFICIENT VALID TRIALS (n={cell_v['n']})")
                        continue
                    real_b, verdict_b = welch_significant(
                        BASELINE_LIOU_RND, cell_v)
                    dmean_b = cell_v['mean'] - BASELINE_LIOU_RND['mean']
                    print(f"    {v} vs baseline LIOU_RND "
                          f"({BASELINE_LIOU_RND['mean']:.4f}+/-"
                          f"{BASELINE_LIOU_RND['std']:.4f},n="
                          f"{BASELINE_LIOU_RND['n']}): "
                          f"dmean={dmean_b:+.4f}  -> {verdict_b}")

        all_results[actual_shape] = shape_results

    # ── Save results ──
    if args.json:
        os.makedirs(os.path.dirname(args.json) or '.', exist_ok=True)
        with open(args.json, 'w') as f:
            json.dump({
                'shapes':      args.shapes,
                'budgets':     args.budgets,
                'variants':    args.variants,
                'trials':      args.trials,
                'total_steps': args.total_steps,
                'K':           LIOU_K,
                'baselines_cow_800F': {
                    '_note':   'from liou_h1_cow.json n=6',
                    'LIOU':    {'mean': 0.9459, 'std': 0.0041},
                    'LIOU_RND':{'mean': 0.9494, 'std': 0.0053},
                    'cc3':     {'mean': 0.9662, 'std': 0.0007},
                },
                'results':     all_results,
            }, f, indent=2)
        print(f"\nRaw results -> {args.json}")


if __name__ == '__main__':
    main()
