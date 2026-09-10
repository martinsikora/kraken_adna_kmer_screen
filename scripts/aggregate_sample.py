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
  <prefix>.damage_model.tsv

With --unit-summaries/--out-stats, additionally aggregates the per-unit
summary TSVs into one per-sample stats row (counts summed, mean_read_length
weighted by classified rows) for downstream use by fit_nnls.
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
    DAMAGE_MODEL_COLUMNS,
    compute_damage_stats,
    fit_damage_models,
    load_damage_arrays,
    load_species_membership_v2,
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
    parser.add_argument("--species-taxids", default="",
                        help="Species taxonomy TSV; used to attach species "
                             "names to the taxid-keyed damage outputs")
    parser.add_argument("--unit-summaries", nargs="+", default=[],
                        help="Per-unit summary.tsv files to aggregate into a "
                             "per-sample stats row")
    parser.add_argument("--out-stats", default="",
                        help="Output path for the aggregated per-sample stats "
                             "TSV (requires --unit-summaries)")

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

    # Model-based damage estimation (emitted alongside damage_score)
    parser.add_argument("--fit-damage-model",    action="store_true", default=True,
                        help="Also fit the per-base damage model by deconvolving "
                             "the k-mer window across read-length strata")
    parser.add_argument("--no-fit-damage-model", action="store_false",
                        dest="fit_damage_model")
    parser.add_argument("--damage-model-min-stratum-reads", type=int, default=70,
                        help="Strata with fewer reads than this do not contribute "
                             "to the model fit")
    return parser.parse_args()


SUMMED_STAT_COLS = [
    "rows_processed", "classified_rows",
    "total_feature_mass", "retained_feature_mass", "unclassified_feature_mass",
    "damage_reads_in_window",
    "n_length_unparseable", "n_malformed_classified_rows",
    "elapsed_seconds",
]


def aggregate_unit_stats(
    summary_paths: list[str],
    merged_vector_nnz: int,
    n_species_accumulated: int,
) -> pd.DataFrame:
    """
    One per-sample stats row from the per-unit summary TSVs.

    Counts and masses are summed; mean_read_length is weighted by classified
    rows; kmer_size is the first nonzero value (disagreements are warned
    about); feature/species counts come from the merged accumulators, not the
    per-unit values, whose supports overlap between lanes.
    """
    frames = [pd.read_csv(p, sep="\t") for p in summary_paths]
    units = pd.concat(frames, ignore_index=True)

    row: dict[str, object] = {"n_units": len(units)}
    for col in SUMMED_STAT_COLS:
        if col in units.columns:
            row[col] = units[col].sum()

    n_classified = float(units["classified_rows"].sum())
    if "mean_read_length" in units.columns and n_classified > 0:
        row["mean_read_length"] = float(
            (units["mean_read_length"] * units["classified_rows"]).sum() / n_classified
        )
    else:
        row["mean_read_length"] = 0.0

    for col in ["damage_min_read_length", "damage_max_read_length"]:
        if col in units.columns:
            vals = units[col].unique()
            if len(vals) > 1:
                print(
                    f"[aggregate_sample] WARNING: {col} differs between units "
                    f"({sorted(vals)}); keeping {vals[0]}",
                    file=sys.stderr, flush=True,
                )
            row[col] = vals[0]

    kmer_sizes = [int(k) for k in units.get("kmer_size", pd.Series(dtype=int)) if int(k) > 0]
    if len(set(kmer_sizes)) > 1:
        print(
            f"[aggregate_sample] WARNING: kmer_size differs between units "
            f"({sorted(set(kmer_sizes))}); keeping {kmer_sizes[0]}",
            file=sys.stderr, flush=True,
        )
    row["kmer_size"] = kmer_sizes[0] if kmer_sizes else 0

    row["n_retained_features"]   = merged_vector_nnz
    row["n_species_accumulated"] = n_species_accumulated
    return pd.DataFrame([row])


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
    if args.species_taxids:
        _, species_id_to_name = load_species_membership_v2(args.species_taxids)
        merged_damage.key_names = species_id_to_name

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

    # --- 4. Model-based damage, alongside damage_score (never replacing it) ---
    if args.fit_damage_model:
        model_df = fit_damage_models(
            merged_damage,
            min_reads         = args.min_reads,
            min_stratum_reads = args.damage_model_min_stratum_reads,
        )
    else:
        model_df = pd.DataFrame(columns=DAMAGE_MODEL_COLUMNS)

    # --- Aggregated per-sample stats from the unit summaries ---
    if args.out_stats:
        if not args.unit_summaries:
            print(
                "[aggregate_sample] ERROR: --out-stats requires --unit-summaries",
                file=sys.stderr,
            )
            sys.exit(1)
        stats_row = aggregate_unit_stats(
            args.unit_summaries,
            merged_vector_nnz     = int(merged_vector.nnz),
            n_species_accumulated = len(merged_damage.taxids()),
        )
        Path(args.out_stats).parent.mkdir(parents=True, exist_ok=True)
        write_tsv(args.out_stats, stats_row)

    prefix = args.out_damage_prefix
    write_tsv(f"{prefix}.damage_profile.tsv",            profile_df)
    write_tsv(f"{prefix}.damage_profile_stratified.tsv", stratified_df)
    write_tsv(f"{prefix}.damage_stats.tsv",              stats_df)
    write_tsv(f"{prefix}.damage_global.tsv",             global_df)
    write_tsv(f"{prefix}.damage_model.tsv",              model_df)

    elapsed = time.perf_counter() - start
    print(
        f"[aggregate_sample] done: elapsed_s={elapsed:.1f} "
        f"n_taxa_profiled={len(global_df)}",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
