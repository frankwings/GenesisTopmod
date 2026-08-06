#!/usr/bin/env python3
"""
acquire_data.py — Download and filter Thingi10K meshes for Phase B distillation.

Writes a manifest JSONL file listing mesh file paths with genus/facet metadata.

Usage:
    python3 acquire_data.py [--cache_dir PATH] [--manifest PATH] [--n_target 2000]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def acquire(args: argparse.Namespace) -> None:
    import thingi10k

    # ── Initialise dataset ─────────────────────────────────────────────
    print(f"[acquire] Initialising thingi10k (cache_dir={args.cache_dir}) …")
    sys.stdout.flush()
    os.makedirs(args.cache_dir, exist_ok=True)
    thingi10k.init(variant='raw', cache_dir=args.cache_dir)

    print("[acquire] Filtering dataset (manifold, closed, single component, "
          "genus 0–2, facets 100–20000) …")
    sys.stdout.flush()
    ds = thingi10k.dataset(
        manifold=True,
        closed=True,
        num_components=1,
        genus=(0, 2),
        num_facets=(100, 20000),
    )

    total_available = len(ds)
    print(f"[acquire] Total available after filtering: {total_available}")
    sys.stdout.flush()

    # ── Shuffle + subsample ────────────────────────────────────────────
    random.seed(42)
    indices = list(range(total_available))
    random.shuffle(indices)
    take_n = min(args.n_target, total_available)
    selected = indices[:take_n]

    # ── Build manifest ─────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.manifest)), exist_ok=True)

    genus_counter: Counter = Counter()
    records = []
    for i, idx in enumerate(selected):
        row = ds[idx]
        entry = {
            'file_id':   int(row['file_id']),
            'file_path': str(row['file_path']),
            'genus':     int(row['genus']),
            'num_facets': int(row['num_facets']),
        }
        records.append(entry)
        genus_counter[int(row['genus'])] += 1

        if (i + 1) % 200 == 0:
            print(f"[acquire] Processed {i+1}/{take_n} …")
            sys.stdout.flush()

    with open(args.manifest, 'w') as f:
        for entry in records:
            f.write(json.dumps(entry) + '\n')

    # ── Stats ──────────────────────────────────────────────────────────
    print(f"\n[acquire] Done.")
    print(f"  Total available  : {total_available}")
    print(f"  Written to manifest: {take_n}")
    print(f"  Manifest path    : {args.manifest}")
    print(f"  Genus distribution:")
    for g in sorted(genus_counter):
        print(f"    genus={g}: {genus_counter[g]}")
    sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Acquire Thingi10K manifest for Phase B")
    parser.add_argument(
        '--cache_dir',
        default=os.path.join(_SCRIPT_DIR, 'data', 'raw'),
        help="Directory for thingi10k raw cache",
    )
    parser.add_argument(
        '--manifest',
        default=os.path.join(_SCRIPT_DIR, 'data', 'manifest.jsonl'),
        help="Output manifest JSONL path",
    )
    parser.add_argument(
        '--n_target',
        type=int,
        default=2000,
        help="Max number of meshes to include in manifest",
    )
    args = parser.parse_args()
    acquire(args)


if __name__ == '__main__':
    main()
