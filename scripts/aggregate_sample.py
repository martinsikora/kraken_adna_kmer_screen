#!/usr/bin/env python3
"""
aggregate_sample.py

Merge per-unit (per-lane) outputs into a single per-sample result.

  1. Sum sparse NNLS feature vectors across all units for the sample
  2. Merge damage accumulator arrays (element-wise addition)
  3. Compute final damage stats from the merged accumulator
  4. Write merged vector + four damage TSV outputs

Outputs written to --out-damage-prefix:
  <prefix>.damage_profile.tsv
  <prefix>.damage_stats.tsv
  <prefix>.damage_global.tsv
  <prefix>.damage_profile_stratified.tsv
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from kraken_screen_lib import (
    compute_damage_stats,
    load_damage_arrays,
    load_sparse_vector,
    merge_damage_accumulators,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-unit outputs into a per-sample result."
    )
    parser.add_argument("--vectors",       nargs="+", required=True,
                        help="Per-unit vector.npz files")
    parser.add_argument("--damage-arrays", nargs="+", required=True,
                        help="Per-unit damage_arrays.npz files (same order as --vectors)")
    parser.add_argument("--out-vector",    required=True,
                        help="Output merged vector npz")
    parser.add_argument("--out-damage-prefix", required=True,
                        help="Prefix for damage output TSVs")

    # Damage scoring parameters
    parser.add_argument("--min-reads",             type=int,   default=100)
    parser.add_argument("--adaptive-plateau",      action="store_true", default=True)
    parser.add_argument("--no-adaptive-plateau",   action="store_false", dest="adaptive_plateau")
    parser.add_argument("--plateau-search-start",  type=int,   default=3)
    parser.add_argument("--plateau-search-end",    type=int,   default=10)
    parser.add_argument("--min-plateau-window",    type=int,   default=3)
    parser.add_argument("--plateau-noise-factor",  type=float, default=2.0)
    parser.add_argument("--plateau-start",         type=int,   default=2,
                        help="Fixed plateau start (used when --no-adaptive-plateau)")
    parser.add_argument("--plateau-end",           type=int,   default=5,
                        help="Fixed plateau end (used when --no-adaptive-plateau)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    # Ensure output directories exist
    Path(args.out_vector).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_damage_prefix).parent.mkdir(parents=True, exist_ok=True)

    if len(args.vectors) != len(args.damage_arrays):
        print(
            f"[aggregate_sample] ERROR: --vectors ({len(args.vectors)}) and "
            f"--damage-arrays ({len(args.damage_arrays)}) must have equal length",
            file=sys.stderr,
        )
        sys.exit(1)

    # --- 1. Sum sparse feature vectors ---
    print(
        f"[aggregate_sample] summing {len(args.vectors)} unit vector(s)",
        file=sys.stderr, flush=True,
    )
    merged_vector = load_sparse_vector(args.vectors[0])
    for path in args.vectors[1:]:
        v = load_sparse_vector(path)
        if v.shape[1] != merged_vector.shape[1]:
            print(
                f"[aggregate_sample] ERROR: vector shape mismatch: "
                f"{v.shape} vs {merged_vector.shape}; cannot aggregate {path}",
                file=sys.stderr,
            )
            sys.exit(1)
        merged_vector = merged_vector + v

    sparse.save_npz(args.out_vector, merged_vector.tocsr())
    print(
        f"[aggregate_sample] merged vector saved: {args.out_vector} "
        f"(nnz={merged_vector.nnz:,})",
        file=sys.stderr, flush=True,
    )

    # --- 2. Merge damage accumulators ---
    print(
        f"[aggregate_sample] merging {len(args.damage_arrays)} damage accumulator(s)",
        file=sys.stderr, flush=True,
    )
    damage_accs = [load_damage_arrays(path) for path in args.damage_arrays]
    merged_damage = merge_damage_accumulators(damage_accs)

    # --- 3. Compute damage stats ---
    stats_df, global_df = compute_damage_stats(
        merged_damage,
        min_reads             = args.min_reads,
        adaptive_plateau      = args.adaptive_plateau,
        plateau_start         = args.plateau_start,
        plateau_end           = args.plateau_end,
        plateau_search_start  = args.plateau_search_start,
        plateau_search_end    = args.plateau_search_end,
        min_plateau_window    = args.min_plateau_window,
        plateau_noise_factor  = args.plateau_noise_factor,
    )
    profile_df       = merged_damage.to_dataframe(min_reads=args.min_reads)
    stratified_df    = merged_damage.to_dataframe_stratified(min_reads=args.min_reads)

    prefix = args.out_damage_prefix
    write_tsv(f"{prefix}.damage_profile.tsv",            profile_df)
    write_tsv(f"{prefix}.damage_profile_stratified.tsv", stratified_df)
    write_tsv(f"{prefix}.damage_stats.tsv",              stats_df)
    write_tsv(f"{prefix}.damage_global.tsv",             global_df)

    elapsed = time.perf_counter() - start
    print(
        f"[aggregate_sample] done: elapsed_s={elapsed:.1f} "
        f"n_taxa_profiled={len(global_df)}",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
