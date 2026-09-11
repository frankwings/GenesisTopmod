#!/usr/bin/env python3
"""
eval_phantom_h2h.py — Head-to-head comparison of Phantom Apex Gradient
                       vs random / heuristic / uniform refinement strategies.

Experiment matrix
-----------------
  Methods : phantom-gradient (PH) / random (RND) / cheap-heuristic (HEUR)
  Budgets : 800F, 960F, 1120F  (from cc2=480F start)
  Shapes  : armadillo (primary), fandisk, dragon  (cow/bunny fallback)
  Trials  : 4  (Welch t-test, noise_floor=0.02)

Hypotheses
----------
  H1: PH > RND at 800F  (phantom knows WHERE to add DOF — causal, not chance)
  H2: PH ≥ HEUR at 960F (causal ≥ correlational signal)
  H3: PH advantage on armadillo > fandisk advantage
      (adaptive refinement has more value when detail density is uneven)

Usage
-----
  # H1 quick check (cow fallback):
  python3 eval_phantom_h2h.py --shapes cow --budgets 800 --trials 4

  # Full matrix:
  python3 eval_phantom_h2h.py --shapes armadillo fandisk dragon \\
      --budgets 800 960 1120 --trials 4 --json eval_out/phantom_h2h.json

  # Via bg_task for long runs:
  python3 -m framework.bg_task launch --id phantom_h1_cow --timeout 3600 \\
      --cwd /home/kingy/Projects/Genesis/GenesisTopmod/experiments/opseq_v5 -- \\
      python3 -u eval_phantom_h2h.py --shapes cow --budgets 800 --trials 4 \\
      --json eval_out/phantom_h1_cow.json
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
    PH_K,
)


def welch_significant(a: dict, b: dict,
                      noise_floor: float = 0.02) -> Tuple[bool, str]:
    """Two-gate significance: |Δmean| > floor AND |Δmean| > combined σ."""
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
                   f"≤ combined σ {combined_sigma:.4f})")
    return real, verdict


def run_one_cell(
    scene: dict, device: str, variant: str,
    budget: int, total_steps: int, trials: int,
) -> dict:
    """Run `trials` repetitions of a variant at a given face budget."""
    ious, faces_list, rounds_list, times = [], [], [], []
    for t in range(trials):
        r = run_variant(variant, scene, device, total_steps,
                        face_budget=budget, ph_K=PH_K)
        ious.append(r['iou'])
        faces_list.append(r['final_faces'])
        rounds_list.append(r.get('n_rounds', 0))
        times.append(r.get('time', 0))
        print(f"    trial {t+1}/{trials}: IoU={r['iou']:.4f}  "
              f"F={r['final_faces']}  rounds={r.get('n_rounds',0)}  "
              f"({r.get('time',0):.0f}s)")
    s = summarize(ious)
    return {
        'iou': s,
        'faces_mean': sum(faces_list) / len(faces_list),
        'rounds_mean': sum(rounds_list) / len(rounds_list),
        'time_mean': sum(times) / len(times),
    }


def main():
    ap = argparse.ArgumentParser(description="Phantom Apex Gradient H2H experiment")
    ap.add_argument('--shapes',      nargs='+', default=['cow'])
    ap.add_argument('--budgets',     nargs='+', type=int, default=[800])
    ap.add_argument('--variants',    nargs='+', default=['PH', 'RND'],
                    choices=['PH', 'RND', 'HEUR'])
    ap.add_argument('--trials',      type=int, default=4)
    ap.add_argument('--total-steps', type=int, default=800)
    ap.add_argument('--device',      default='cuda')
    ap.add_argument('--json',        default=None)
    ap.add_argument('--h1-only',     action='store_true',
                    help='Run only H1 test (PH vs RND at first budget)')
    args = ap.parse_args()

    device = args.device if torch.cuda.is_available() else 'cpu'

    if args.h1_only:
        args.variants = ['PH', 'RND']
        args.budgets  = args.budgets[:1]

    print("=" * 72)
    print(f"PHANTOM APEX GRADIENT H2H EXPERIMENT")
    print(f"  shapes   = {args.shapes}")
    print(f"  budgets  = {args.budgets}")
    print(f"  variants = {args.variants}")
    print(f"  trials   = {args.trials}")
    print(f"  steps    = {args.total_steps}")
    print(f"  K        = {PH_K}")
    print("=" * 72)

    all_results: Dict[str, dict] = {}

    for shape in args.shapes:
        print(f"\n{'─' * 72}")
        print(f"SHAPE: {shape}")
        print(f"{'─' * 72}")

        scene = setup_scene(shape, device)
        actual_shape = scene['shape_name']
        if actual_shape != shape:
            print(f"  WARNING: {shape} not found, using {actual_shape}")

        shape_results: Dict[str, Dict[str, dict]] = {}

        for budget in args.budgets:
            print(f"\n  ── Budget {budget}F ──")
            budget_results: Dict[str, dict] = {}

            for v in args.variants:
                print(f"\n  --- {v} × {args.trials} @ {budget}F ---")
                cell = run_one_cell(
                    scene, device, v, budget, args.total_steps, args.trials)
                budget_results[v] = cell
                s = cell['iou']
                print(f"  → {v}: IoU={s['mean']:.4f} ± {s['std']:.4f}  "
                      f"[{s['min']:.4f}, {s['max']:.4f}]  "
                      f"F_mean={cell['faces_mean']:.0f}  "
                      f"rounds={cell['rounds_mean']:.1f}  "
                      f"time={cell['time_mean']:.0f}s")

            shape_results[str(budget)] = budget_results

            # Pairwise comparisons for this budget
            print(f"\n  PAIRWISE @ {budget}F (noise_floor=0.02):")
            vs = args.variants
            for i in range(len(vs)):
                for j in range(i + 1, len(vs)):
                    a = budget_results[vs[i]]['iou']
                    b = budget_results[vs[j]]['iou']
                    real, verdict = welch_significant(a, b)
                    dmean = b['mean'] - a['mean']
                    print(f"    {vs[j]} vs {vs[i]}: Δmean={dmean:+.4f}  → {verdict}")

        all_results[actual_shape] = shape_results

    # ── Hypothesis testing ──
    print("\n" + "=" * 72)
    print("HYPOTHESIS TESTS")
    print("=" * 72)

    for shape in all_results:
        sr = all_results[shape]

        # H1: PH > RND at first budget
        first_budget = str(args.budgets[0])
        if first_budget in sr and 'PH' in sr[first_budget] and 'RND' in sr[first_budget]:
            ph = sr[first_budget]['PH']['iou']
            rnd = sr[first_budget]['RND']['iou']
            real, verdict = welch_significant(rnd, ph)
            dmean = ph['mean'] - rnd['mean']
            h1_pass = real and dmean > 0
            print(f"\n  H1 [{shape} @ {first_budget}F]: PH > RND?")
            print(f"    PH:  {ph['mean']:.4f} ± {ph['std']:.4f}")
            print(f"    RND: {rnd['mean']:.4f} ± {rnd['std']:.4f}")
            print(f"    Δmean = {dmean:+.4f}  → {verdict}")
            print(f"    H1 {'PASS ✓' if h1_pass else 'FAIL ✗'}")
            if not h1_pass:
                print(f"    ⚠ H1 FAILED — phantom gradient does NOT significantly "
                      f"beat random at {first_budget}F. Consider fallback to "
                      f"cheap heuristic.")

        # H2: PH ≥ HEUR at 960F
        if '960' in sr and 'PH' in sr['960'] and 'HEUR' in sr['960']:
            ph = sr['960']['PH']['iou']
            heur = sr['960']['HEUR']['iou']
            real, verdict = welch_significant(heur, ph)
            dmean = ph['mean'] - heur['mean']
            h2_pass = dmean >= 0 or (not real)  # PH ≥ HEUR or indistinguishable
            print(f"\n  H2 [{shape} @ 960F]: PH ≥ HEUR?")
            print(f"    PH:   {ph['mean']:.4f} ± {ph['std']:.4f}")
            print(f"    HEUR: {heur['mean']:.4f} ± {heur['std']:.4f}")
            print(f"    Δmean = {dmean:+.4f}  → {verdict}")
            print(f"    H2 {'PASS ✓' if h2_pass else 'FAIL ✗'}")

    # H3: cross-shape comparison
    shapes_done = list(all_results.keys())
    if len(shapes_done) >= 2 and args.budgets:
        budget_str = str(args.budgets[0])
        print(f"\n  H3: Cross-shape phantom advantage comparison @ {budget_str}F:")
        advantages = {}
        for shape in shapes_done:
            sr = all_results[shape]
            if (budget_str in sr and 'PH' in sr[budget_str]
                    and 'RND' in sr[budget_str]):
                ph  = sr[budget_str]['PH']['iou']['mean']
                rnd = sr[budget_str]['RND']['iou']['mean']
                adv = ph - rnd
                advantages[shape] = adv
                print(f"    {shape}: PH−RND = {adv:+.4f}")
        if len(advantages) >= 2:
            best = max(advantages, key=advantages.get)
            worst = min(advantages, key=advantages.get)
            print(f"    Largest advantage: {best} ({advantages[best]:+.4f})")
            print(f"    Smallest advantage: {worst} ({advantages[worst]:+.4f})")

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
                'K':           PH_K,
                'results':     all_results,
            }, f, indent=2)
        print(f"\nRaw results → {args.json}")


if __name__ == '__main__':
    main()
