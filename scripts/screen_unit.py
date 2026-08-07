#!/usr/bin/env python3
"""
screen_unit.py

Single-pass analysis of one KrakenUniq classify file (.tsv.gz).

In one streaming pass through the (potentially large) compressed input:
  1. Builds a sparse feature vector for NNLS abundance estimation
  2. Accumulates aDNA damage profiles (absolute and fractional position)

Outputs:
  --out-vector        {unit_id}.vector.npz           sparse NNLS feature vector
  --out-damage-arrays {unit_id}.damage_arrays.npz    serialised accumulator state
  --out-summary       {unit_id}.summary.tsv          per-unit processing stats
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from kraken_screen_lib import (
    build_feature_lookup,
    end_flags_from_runs,
    load_species_membership_v2,
    parse_kmer_string_runs,
    parse_strata_spec,
    save_sparse_vector,
    save_damage_arrays,
    write_tsv,
    DamageAccumulator,
    FractionalAccumulator,
    open_buffered_gzip,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-pass KrakenUniq classify file analysis: vectorize + damage accumulation."
    )
    parser.add_argument("--kraken-class",       required=True,
                        help="KrakenUniq classify TSV (plain or .gz)")
    parser.add_argument("--reference-features", required=True,
                        help="Reference feature index TSV (feature_index, taxid)")
    parser.add_argument("--species-taxids",     required=True,
                        help="Species taxonomy TSV (tax_rank, tax_id, tax_name, tax_ids_descendant)")
    parser.add_argument("--out-vector",         required=True)
    parser.add_argument("--out-damage-arrays",  required=True)
    parser.add_argument("--out-summary",        required=True)
    parser.add_argument("--exclude-taxids",     nargs="+", type=int, default=[0, 1, 2, 131567],
                        help="Taxids to exclude from feature vector (always includes 0)")
    parser.add_argument("--max-pos",            type=int, default=25)
    parser.add_argument("--n-bins",             type=int, default=100)
    parser.add_argument("--strata",             nargs="+", default=["31-40", "41-55", "56-75", "76-100"])
    parser.add_argument("--progress-every",     type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()

    # Ensure output directories exist
    for p in [args.out_vector, args.out_damage_arrays, args.out_summary]:
        Path(p).parent.mkdir(parents=True, exist_ok=True)

    # Load reference feature lookup: feature_taxid -> feature_index
    feature_lookup = build_feature_lookup(args.reference_features)
    n_features = max(feature_lookup.values()) + 1 if feature_lookup else 0
    feature_taxid_set: set[int] = set(feature_lookup.keys())

    # Load species membership: child_taxid -> (species_taxid, species_name)
    child_to_species, _ = load_species_membership_v2(args.species_taxids)

    # Build exclusion set (always include 0=unclassified for damage tracking,
    # but exclude from feature vector accumulation)
    exclude_set: set[int] = set(args.exclude_taxids)
    exclude_set.add(0)

    # Initialise accumulators
    strata = parse_strata_spec(args.strata)
    damage_acc = DamageAccumulator(max_pos=args.max_pos)
    frac_acc   = FractionalAccumulator(strata=strata, n_bins=args.n_bins)

    # Feature vector accumulation
    feature_counts: dict[int, float] = {}

    # Stats
    n_rows = 0
    n_classified = 0
    total_mass = 0.0
    retained_mass = 0.0
    unclassified_mass = 0.0
    n_species_accumulated: set[str] = set()

    print(
        f"[screen_unit] starting: {args.kraken_class}",
        file=sys.stderr, flush=True,
    )

    with open_buffered_gzip(args.kraken_class) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) < 5:
                continue
            n_rows += 1

            status, read_id, taxid_str, length_str, kmer_str = parts

            # Parse kmer string once for both purposes, keeping run-length form
            tids, counts, taxid_counts, nk, unc = parse_kmer_string_runs(
                kmer_str, exclude_set
            )

            # Track all parsed k-mer mass (including excluded taxa and "A" tokens)
            total_mass += float(nk)
            unclassified_mass += float(unc)

            # --- NNLS vector path ---
            for feat_taxid, count in taxid_counts.items():
                feat_idx = feature_lookup.get(feat_taxid)
                if feat_idx is None:
                    continue
                feature_counts[feat_idx] = feature_counts.get(feat_idx, 0.0) + count
                retained_mass += count

            # --- Damage path (classified reads only) ---
            if status == "C":
                n_classified += 1
                try:
                    taxid = int(taxid_str)
                    read_len = int(length_str)
                except ValueError:
                    continue
                species_info = child_to_species.get(taxid)
                if species_info is not None:
                    _, species_name = species_info
                    if nk > 0:
                        n5 = min(nk, args.max_pos)
                        head, tail = end_flags_from_runs(tids, counts, n5)
                        damage_acc.add_flags(species_name, n5, head, tail)
                        frac_acc.add_runs(species_name, read_len, tids, counts,
                                          nk, unc > 0)
                        n_species_accumulated.add(species_name)

            if args.progress_every > 0 and n_rows % args.progress_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[screen_unit] rows={n_rows:,} elapsed_s={elapsed:.1f} "
                    f"classified={n_classified:,} species={len(n_species_accumulated):,}",
                    file=sys.stderr, flush=True,
                )

    elapsed = time.perf_counter() - start
    print(
        f"[screen_unit] done: rows={n_rows:,} elapsed_s={elapsed:.1f} "
        f"features={len(feature_counts):,} species={len(n_species_accumulated):,}",
        file=sys.stderr, flush=True,
    )

    # Save sparse vector
    if feature_counts:
        indices = np.array(list(feature_counts.keys()),  dtype=np.int32)
        data    = np.array(list(feature_counts.values()), dtype=np.float64)
    else:
        indices = np.zeros(0, dtype=np.int32)
        data    = np.zeros(0, dtype=np.float64)

    save_sparse_vector(args.out_vector, indices, data, n_features)

    # Save damage accumulator state
    save_damage_arrays(args.out_damage_arrays, damage_acc, frac_acc)

    # Save summary
    write_tsv(
        args.out_summary,
        pd.DataFrame([{
            "rows_processed":           n_rows,
            "classified_rows":          n_classified,
            "total_feature_mass":       total_mass,
            "retained_feature_mass":    retained_mass,
            "unclassified_feature_mass": unclassified_mass,
            "n_retained_features":      len(feature_counts),
            "n_species_accumulated":    len(n_species_accumulated),
            "elapsed_seconds":          elapsed,
        }]),
    )


if __name__ == "__main__":
    main()
