#!/usr/bin/env python3
"""
screen_unit.py

Single-pass analysis of one KrakenUniq classify file (.tsv.gz).

In one streaming pass through the (potentially large) compressed input:
  1. Builds a sparse feature vector for NNLS abundance estimation
  2. Accumulates aDNA damage profiles by position from each read end,
     pooled and split by read-length stratum

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
    parser.add_argument("--strata",             nargs="+", default=["31-40", "41-55", "56-75", "76-100"])
    parser.add_argument("--min-read-length",    type=int, default=30,
                        help="Reads shorter than this are excluded from damage "
                             "accumulation (abundance is unaffected)")
    parser.add_argument("--max-read-length",    type=int, default=75,
                        help="Reads longer than this are excluded from damage "
                             "accumulation. Set below the shortest sequencing read "
                             "length in the run so that every retained read is a "
                             "complete molecule: a read at the read-length cap has "
                             "been truncated, so its 3' end is a sequencing cut-off "
                             "and carries no terminal damage")
    parser.add_argument("--progress-every",     type=int, default=1_000_000)
    parser.add_argument("--kmer-size",          type=int, default=0,
                        help="Database k-mer size. 0 (default) infers it as the "
                             "modal value of length - n_kmers + 1 over the first "
                             "1000 parseable rows")
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
    child_to_species, species_id_to_name = load_species_membership_v2(args.species_taxids)

    # Build exclusion set (always include 0=unclassified for damage tracking,
    # but exclude from feature vector accumulation)
    exclude_set: set[int] = set(args.exclude_taxids)
    exclude_set.add(0)

    # Initialise accumulators
    strata = parse_strata_spec(args.strata)
    damage_acc = DamageAccumulator(max_pos=args.max_pos, strata=strata,
                                   kmer_size=max(args.kmer_size, 0))
    damage_acc.key_names = species_id_to_name

    # k-mer size inference state: k = length - n_kmers + 1 for any read.
    # Collect observations over a warm-up window and take the modal value,
    # instead of trusting whichever read happens to parse first.
    kmer_obs: dict[int, int] = {}
    kmer_warmup_left = 1000 if damage_acc.kmer_size == 0 else 0
    n_length_unparseable = 0

    def finalize_kmer_size() -> None:
        if damage_acc.kmer_size == 0 and kmer_obs:
            mode_k = max(kmer_obs.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            if len(kmer_obs) > 1:
                print(
                    f"[screen_unit] WARNING: inconsistent inferred k-mer sizes "
                    f"{dict(sorted(kmer_obs.items()))}; using modal value {mode_k}",
                    file=sys.stderr, flush=True,
                )
            damage_acc.kmer_size = mode_k

    # Feature vector accumulation
    feature_counts: dict[int, float] = {}

    # Stats
    n_rows = 0
    n_classified = 0
    total_mass = 0.0
    retained_mass = 0.0
    unclassified_mass = 0.0
    n_species_accumulated: set[int] = set()
    n_damage_reads = 0
    n_malformed_classified_rows = 0
    max_read_len_seen = 0
    # Summed over every classified read, not only the damage window: the
    # coverage statistics this feeds are computed from all classified reads.
    # coverage_evenness turns it into a depth estimate taken from bases
    # sequenced (reads * length / genome length) for its evenness_depth column.
    read_length_sum = 0

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

            if kmer_warmup_left > 0 and nk > 0:
                try:
                    k_obs = int(length_str) - nk + 1
                except ValueError:
                    n_length_unparseable += 1
                else:
                    if k_obs > 0:
                        kmer_obs[k_obs] = kmer_obs.get(k_obs, 0) + 1
                        kmer_warmup_left -= 1
                        if kmer_warmup_left == 0:
                            finalize_kmer_size()

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
                try:
                    taxid = int(taxid_str)
                    read_len = int(length_str)
                except ValueError:
                    # e.g. paired-end "75|75" length fields
                    n_malformed_classified_rows += 1
                    continue
                n_classified += 1
                read_length_sum += read_len
                if read_len > max_read_len_seen:
                    max_read_len_seen = read_len
                # Damage is estimated only from reads within the length window.
                # Reads at the sequencing read-length cap are truncated molecules
                # whose 3' end is not a molecule terminus, and pooling lanes with
                # different read lengths otherwise mixes different cap positions
                # into one profile.
                if not (args.min_read_length <= read_len <= args.max_read_length):
                    continue
                species_info = child_to_species.get(taxid)
                if species_info is not None:
                    species_taxid, _ = species_info
                    if nk > 0:
                        n_damage_reads += 1
                        n5 = min(nk, args.max_pos)
                        head, tail = end_flags_from_runs(tids, counts, n5)
                        damage_acc.add_flags(
                            species_taxid, n5, head, tail,
                            stratum_idx=damage_acc.stratum_index(read_len),
                        )
                        n_species_accumulated.add(species_taxid)

            if args.progress_every > 0 and n_rows % args.progress_every == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[screen_unit] rows={n_rows:,} elapsed_s={elapsed:.1f} "
                    f"classified={n_classified:,} species={len(n_species_accumulated):,}",
                    file=sys.stderr, flush=True,
                )

    finalize_kmer_size()

    if damage_acc.kmer_size == 0 and n_rows > 0:
        if n_length_unparseable > 0 and not kmer_obs:
            print(
                f"[screen_unit] ERROR: k-mer size could not be inferred — every "
                f"sampled row has an unparseable length field "
                f"({n_length_unparseable} rows, e.g. paired-end '75|75'). "
                f"All damage statistics would be silently empty. "
                f"Pass --kmer-size explicitly if this input is genuinely usable.",
                file=sys.stderr, flush=True,
            )
            sys.exit(1)
        print(
            "[screen_unit] WARNING: k-mer size could not be inferred "
            "(no rows with parseable length and k-mers); the damage model "
            "will be skipped downstream. Pass --kmer-size to set it explicitly.",
            file=sys.stderr, flush=True,
        )

    if n_classified > 0 and max_read_len_seen <= args.max_read_length:
        print(
            f"[screen_unit] WARNING: longest classified read ({max_read_len_seen}) "
            f"does not exceed --max-read-length ({args.max_read_length}), so the "
            f"damage read-length cap never excludes anything. This suggests the "
            f"cap sits at or above the sequencing read length; reads at the "
            f"sequencing cap are truncated molecules whose 3' ends contaminate "
            f"the damage profile — set damage_max_read_length strictly below "
            f"the sequencing read length.",
            file=sys.stderr, flush=True,
        )

    elapsed = time.perf_counter() - start
    print(
        f"[screen_unit] done: rows={n_rows:,} elapsed_s={elapsed:.1f} "
        f"features={len(feature_counts):,} species={len(n_species_accumulated):,} "
        f"damage_reads={n_damage_reads:,} "
        f"(length {args.min_read_length}-{args.max_read_length}, "
        f"k={damage_acc.kmer_size})",
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
    save_damage_arrays(args.out_damage_arrays, damage_acc)

    # Save summary
    write_tsv(
        args.out_summary,
        pd.DataFrame([{
            "rows_processed":           n_rows,
            # Classified rows with parseable taxid and length fields; rows a
            # malformed field excluded are counted separately below.
            "classified_rows":          n_classified,
            "total_feature_mass":       total_mass,
            "retained_feature_mass":    retained_mass,
            "unclassified_feature_mass": unclassified_mass,
            "n_retained_features":      len(feature_counts),
            "n_species_accumulated":    len(n_species_accumulated),
            "mean_read_length":         (read_length_sum / n_classified
                                         if n_classified else 0.0),
            "damage_reads_in_window":   n_damage_reads,
            "damage_min_read_length":   args.min_read_length,
            "damage_max_read_length":   args.max_read_length,
            "kmer_size":                damage_acc.kmer_size,
            "n_length_unparseable":     n_length_unparseable,
            "n_malformed_classified_rows": n_malformed_classified_rows,
            "elapsed_seconds":          elapsed,
        }]),
    )


if __name__ == "__main__":
    main()
