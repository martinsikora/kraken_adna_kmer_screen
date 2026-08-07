#!/usr/bin/env python3
"""
build_species_genome_lengths.py

Derive a per-species genome length from a KrakenUniq database's
library_seq_info.tsv, for the depth-normalised evenness score in
coverage_evenness.py. Run once per database, alongside build_reference_matrix.

Why not use the KrakenUniq report directly: the report's `cov` is breadth
against the union of taxon-discriminative k-mers across every strain of that
species in the database. That union grows as a species gets better sequenced,
so `cov` is deflated for well-represented taxa -- Hepatitis B virus, with ~9950
strains, has a ~1.3 Mb denominator for a 3.2 kb genome. A genome length taken
from the library lets the depth of coverage be estimated from read count alone,
independently of `dup`, which is what the evenness score needs.

Why median-of-assembly-sums: a species' sequences span several assemblies, and
each assembly spans several replicons (chromosome plus plasmids). Summing every
sequence for a species multiplies the genome by the number of assemblies;
averaging sequence lengths divides it by the number of replicons (Yersinia
pestis comes out at 1.30 Mb against a true 4.65 Mb, because its three plasmids
drag the mean down). Summing within an assembly and taking the median across
assemblies avoids both.

Validated against published genome sizes: Streptococcus mutans 2.030 Mb (ratio
1.000), Hepatitis B virus 3215 bp (1.005), Yersinia enterocolitica 4.675 Mb
(1.016), Clostridium sporogenes 4.149 Mb (1.012), Yersinia pestis 4.784 Mb
(1.029), Ralstonia insidiosa 6.00 Mb (1.072).

Usage:
    python scripts/build_species_genome_lengths.py \
        --library-seq-info /path/to/krakendb/library_seq_info.tsv \
        --out /path/to/krakendb/kraken_adna_kmer_screen_db/species.genome_lengths.tsv
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

REQUIRED_COLS = [
    "assembly_id", "tax_id_species", "tax_name_species",
    "seq_l_tot", "seq_l_masked",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-species genome length from a KrakenUniq library_seq_info.tsv"
    )
    p.add_argument("--library-seq-info", required=True,
                   help="library_seq_info.tsv from the KrakenUniq database directory")
    p.add_argument("--out", required=True,
                   help="Output TSV: tax_id_species, tax_name_species, "
                        "n_assemblies, genome_length, genome_length_masked")
    p.add_argument("--min-assemblies", type=int, default=1,
                   help="Skip species with fewer assemblies than this")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.library_seq_info, sep="\t", usecols=REQUIRED_COLS)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        sys.exit(f"[genome_lengths] ERROR: missing columns: {missing}")

    df = df.dropna(subset=["tax_id_species", "assembly_id"])
    print(
        f"[genome_lengths] {len(df):,} sequences, "
        f"{df['assembly_id'].nunique():,} assemblies, "
        f"{df['tax_id_species'].nunique():,} species",
        file=sys.stderr, flush=True,
    )

    # sum replicons within an assembly, then take the median assembly per species
    per_assembly = (
        df.groupby(["tax_id_species", "tax_name_species", "assembly_id"],
                   observed=True)[["seq_l_tot", "seq_l_masked"]]
          .sum()
          .reset_index()
    )
    per_species = (
        per_assembly.groupby(["tax_id_species", "tax_name_species"], observed=True)
        .agg(n_assemblies=("assembly_id", "nunique"),
             genome_length=("seq_l_tot", "median"),
             genome_length_masked=("seq_l_masked", "median"))
        .reset_index()
    )

    if args.min_assemblies > 1:
        per_species = per_species[per_species["n_assemblies"] >= args.min_assemblies]

    per_species["tax_id_species"] = per_species["tax_id_species"].astype("Int64")
    for c in ("genome_length", "genome_length_masked"):
        per_species[c] = per_species[c].round().astype("Int64")
    per_species = per_species.sort_values("tax_name_species").reset_index(drop=True)

    per_species.to_csv(args.out, sep="\t", index=False)
    print(
        f"[genome_lengths] wrote {len(per_species):,} species -> {args.out} "
        f"(median genome {per_species['genome_length'].median():,.0f} bp)",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
