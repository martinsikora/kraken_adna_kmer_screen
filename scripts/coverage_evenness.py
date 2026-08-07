#!/usr/bin/env python3
"""
coverage_evenness.py

Parse one or more KrakenUniq report TSV files, merge lanes, and compute
per-taxon coverage evenness statistics.

Evenness index (Lander-Waterman ratio):
  E = cov / (1 - exp(-dup * cov))

  where:
    cov = unique_kmers / total_genome_kmers_in_db  (breadth of coverage)
    dup = total_kmers / unique_kmers               (mean depth per covered position)
    dup * cov = average depth per genome position

  E ≈ 1 : coverage consistent with Poisson (uniform read placement)
  E < 1 : reads are clumped (observed breadth < Poisson expectation)

A second, depth-normalised evenness score is emitted alongside it when a
species genome-length table and the per-unit summaries are supplied:

  evenness_depth = cov / (1 - exp(-depth_estimate))
  depth_estimate = reads * mean_read_length / genome_length

What this does and does not change. evenness_index is a correct Lander-Waterman
ratio, and its reduction to 1/dup at low depth is the right answer there: with
N k-mer observations spread over G positions, expected breadth is N/G, observed
is unique/G, and the ratio is unique/N = 1/dup. The denominator G cancels, so
ANY consistently normalised variant gives the same number. That was tested by
renormalising with the database's own per-species k-mer counts
(database.kdb.counts, whose clade sums reproduce kmers/cov exactly): the result
matched evenness_index to seven decimal places, Spearman 1.000000, max absolute
difference 9.6e-07. Changing the denominator changes nothing.

evenness_depth differs for one reason only: its depth comes from bases
sequenced (reads * mean_read_length) rather than from k-mer observations
(dup * kmers). That is deliberately an inconsistent normalisation -- a bases
numerator against a discriminative-k-mer denominator -- and it is where the
behaviour comes from, not from any improvement in the depth estimate.

So what it measures is closer to unique discriminative k-mers per base
sequenced: 1/dup scaled by the fraction of each read's k-mers that discriminate
the taxon. That fraction is small for taxa with close relatives in the database,
which are the ones prone to misassignment, so the score blends coverage evenness
with taxonomic distinctiveness. For screening that blend behaves better than
evenness alone. Against interior_rate from the damage model (a proxy for reads
not really belonging to the taxon), Spearman is -0.197 on 018345 (n=5208) and
-0.160 on DA195 (n=2252), while evenness_index (+0.042 / +0.007), cov and dup
are all uncorrelated and two of them carry the wrong sign. Cross-sample spread
in pass rate is 8.9-fold against 1884-fold for evenness_index > 0.5.

Read it as a screening statistic, not as a pure evenness measure.

evenness_depth is emitted as an extra column and is not used by any hit
criterion; evenness_index remains the criterion.

Output columns (all snake_case):
  sample_id, tax_id, rank, tax_name, reads, tax_reads, kmers,
  dup, cov, evenness_index, genome_length, depth_estimate, evenness_depth,
  kmer_set_ratio

Caveat carried by kmer_set_ratio: cov is breadth against the union of
taxon-discriminative k-mers over every strain in the database, not against the
genome, so evenness_depth pairs a cov numerator with a genome denominator.
kmer_set_ratio = (kmers/cov) / genome_length reports how far apart the two are.
Near 1 the score is sound; far from 1 it is not. Hepatitis B virus sits at ~400
(9950 strains), and its evenness_depth reads 0.004 despite 12x coverage.
Yersinia pestis sits at ~0.13, most of its genome being shared with
Y. pseudotuberculosis and assigned above the species node. Filter on
kmer_set_ratio before using evenness_depth.

This cannot be fixed from the database as it stands. It would need the number
of discriminative k-mers in a single representative genome, and the per-taxon
counts do not carry it: k-mers shared across strains sit at the species node
while strain nodes hold only strain-specific k-mers, so the per-strain counts
are tiny and unrelated to genome size (Hepatitis B virus 64, Yersinia pestis
207). Recovering it would mean re-deriving k-mer sets per genome from the
database itself.
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kraken_screen_lib import write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute per-taxon coverage evenness from KrakenUniq report TSVs."
    )
    parser.add_argument("--reports",      nargs="+", required=True,
                        help="KrakenUniq report TSV files (one or more lanes)")
    parser.add_argument("--out-coverage", required=True)
    parser.add_argument("--sample-id",    required=True)
    parser.add_argument("--min-reads",    type=int, default=50)
    parser.add_argument("--ranks",        nargs="+",
                        default=["species", "genus"],
                        help="Taxonomy ranks to include in output (e.g. species genus)")
    parser.add_argument("--species-genome-lengths", default=None,
                        help="TSV from build_species_genome_lengths.py. Enables "
                             "the depth-normalised evenness_depth column; "
                             "without it that column is written as NaN")
    parser.add_argument("--unit-summaries", nargs="*", default=None,
                        help="Per-unit summary.tsv files from screen_unit, used "
                             "for the sample's mean classified read length")
    return parser.parse_args()


def load_genome_lengths(path: str | None) -> dict:
    """species tax_id -> genome length in bp. Empty dict when unavailable."""
    if not path:
        return {}
    try:
        g = pd.read_csv(path, sep="\t")
    except Exception as exc:
        print(f"[coverage_evenness] WARNING: could not read {path}: {exc}",
              file=sys.stderr)
        return {}
    col = "genome_length_masked" if "genome_length_masked" in g.columns else "genome_length"
    if "tax_id_species" not in g.columns or col not in g.columns:
        print(f"[coverage_evenness] WARNING: {path} lacks expected columns",
              file=sys.stderr)
        return {}
    g = g.dropna(subset=["tax_id_species", col])
    return dict(zip(g["tax_id_species"].astype("int64"),
                    pd.to_numeric(g[col], errors="coerce").astype(float)))


def mean_read_length(paths: list | None) -> float:
    """
    Read-weighted mean classified read length across a sample's units.

    Returns nan when the summaries are missing or predate the mean_read_length
    column, which disables evenness_depth rather than guessing a length.
    """
    if not paths:
        return float("nan")
    tot_len = tot_n = 0.0
    for p in paths:
        try:
            d = pd.read_csv(p, sep="\t")
        except Exception:
            continue
        if "mean_read_length" not in d.columns or "classified_rows" not in d.columns:
            continue
        n = pd.to_numeric(d["classified_rows"], errors="coerce").fillna(0).sum()
        m = pd.to_numeric(d["mean_read_length"], errors="coerce").fillna(0).sum()
        tot_len += float(m) * float(n)
        tot_n += float(n)
    return tot_len / tot_n if tot_n > 0 else float("nan")


# ---------------------------------------------------------------------------
# Report parsing
# ---------------------------------------------------------------------------

REPORT_COLS = ["pct_reads", "reads", "tax_reads", "kmers", "dup", "cov", "tax_id", "rank", "tax_name"]


def open_maybe_gzip(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "rt", encoding="utf-8")


def parse_kraken_report(path: str) -> pd.DataFrame:
    """
    Parse a single KrakenUniq report TSV into a flat DataFrame.

    Skips ## comment lines and the header line. Strips leading whitespace
    from tax_name (KrakenUniq indents names hierarchically).
    """
    rows = []
    with open_maybe_gzip(path) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9:
                continue
            # Header line detection (first column is '%')
            if parts[0].strip() == "%":
                continue
            try:
                row = {
                    "pct_reads": float(parts[0]),
                    "reads":     int(parts[1]),
                    "tax_reads": int(parts[2]),
                    "kmers":     int(parts[3]),
                    "dup":       float(parts[4]) if parts[4].strip().upper() not in ("NA", "N/A", "") else np.nan,
                    "cov":       float(parts[5]) if parts[5].strip().upper() not in ("NA", "N/A", "") else np.nan,
                    "tax_id":    int(parts[6]),
                    "rank":      parts[7].strip(),
                    "tax_name":  parts[8].strip(),
                }
            except (ValueError, IndexError):
                continue
            rows.append(row)
    if not rows:
        return pd.DataFrame(columns=REPORT_COLS)
    return pd.DataFrame(rows)


def merge_reports(report_dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Merge multiple per-lane reports by summing reads and re-deriving dup/cov.

    For dup (= total_kmers / unique_kmers):
      - Derive total_kmers = kmers * dup per lane and sum across lanes.
      - Estimate merged unique_kmers from merged cov and estimated genome_kmers.

    For cov (= unique_kmers / genome_kmers_in_db):
      - Use overlap-aware probabilistic union:
          cov_union = 1 - product(1 - cov_lane)
        This avoids systematic overestimation from simple summation.
    """
    if len(report_dfs) == 1:
        return report_dfs[0].copy()

    dfs = []
    for df in report_dfs:
        d = df.copy()
        d["total_kmers"] = d["kmers"] * d["dup"]
        with np.errstate(divide="ignore", invalid="ignore"):
            d["genome_kmers_est"] = np.where(d["cov"] > 0, d["kmers"] / d["cov"], np.nan)
        dfs.append(d)

    combined = pd.concat(dfs, ignore_index=True)
    grouped = combined.groupby(["tax_id", "rank", "tax_name"], sort=False, dropna=False)

    out_rows = []
    for (tax_id, rank, tax_name), g in grouped:
        reads = int(g["reads"].sum())
        tax_reads = int(g["tax_reads"].sum())
        total_kmers = float(np.nansum(g["total_kmers"].to_numpy(dtype=float)))

        cov_vals = g["cov"].to_numpy(dtype=float)
        cov_valid = cov_vals[np.isfinite(cov_vals)]
        cov_valid = np.clip(cov_valid, 0.0, 1.0)
        if cov_valid.size > 0:
            cov_union = float(1.0 - np.prod(1.0 - cov_valid))
        else:
            cov_union = np.nan

        genome_vals = g["genome_kmers_est"].to_numpy(dtype=float)
        genome_valid = genome_vals[np.isfinite(genome_vals) & (genome_vals > 0)]
        if genome_valid.size > 0 and np.isfinite(cov_union):
            genome_kmers = float(np.median(genome_valid))
            kmers_merged = max(0.0, min(genome_kmers * cov_union, genome_kmers))
        else:
            # Fallback when cov/genome estimates are unavailable:
            # use max lane unique-kmers (conservative under overlap).
            kmers_merged = float(np.nanmax(g["kmers"].to_numpy(dtype=float)))

        dup_merged = total_kmers / kmers_merged if kmers_merged > 0 else np.nan

        out_rows.append({
            "tax_id": int(tax_id),
            "rank": str(rank),
            "tax_name": str(tax_name),
            "pct_reads": np.nan,
            "reads": reads,
            "tax_reads": tax_reads,
            "kmers": int(round(kmers_merged)) if np.isfinite(kmers_merged) else 0,
            "dup": float(dup_merged) if np.isfinite(dup_merged) else np.nan,
            "cov": float(cov_union) if np.isfinite(cov_union) else np.nan,
        })

    return pd.DataFrame(out_rows, columns=REPORT_COLS)


# ---------------------------------------------------------------------------
# Evenness computation
# ---------------------------------------------------------------------------

def compute_evenness(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add evenness_index column using the Lander-Waterman ratio.

    E = cov / (1 - exp(-dup * cov))

    Special cases:
      - dup or cov is NaN  → evenness_index = NaN
      - cov == 0           → evenness_index = NaN
      - |1 - exp(-lc)| < 1e-10 (near-zero denominator) → evenness_index = NaN
    """
    d = df.copy()
    dup = d["dup"].to_numpy(dtype=float)
    cov = d["cov"].to_numpy(dtype=float)
    lc  = dup * cov
    denominator = 1.0 - np.exp(-lc)
    safe_denom  = np.where(np.abs(denominator) < 1e-10, np.nan, denominator)
    d["evenness_index"] = np.where(
        (np.isfinite(dup) & np.isfinite(cov) & (cov > 0)),
        cov / safe_denom,
        np.nan,
    )
    return d


def compute_evenness_depth(
    df: pd.DataFrame, genome_lengths: dict, read_len: float,
) -> pd.DataFrame:
    """
    Add genome_length, depth_estimate and evenness_depth.

    depth_estimate = reads * mean_read_length / genome_length. Taking depth from
    bases sequenced rather than from k-mer observations is what makes this
    differ from evenness_index at all: a consistently normalised ratio cancels
    its denominator and returns 1/dup whatever it is normalised by. See the
    module docstring for what the resulting score measures. Columns are NaN
    wherever the genome length or the read length is unavailable.
    """
    d = df.copy()
    n = len(d)
    if not genome_lengths or not np.isfinite(read_len) or read_len <= 0:
        d["genome_length"] = np.nan
        d["depth_estimate"] = np.nan
        d["evenness_depth"] = np.nan
        d["kmer_set_ratio"] = np.nan
        return d

    tax = pd.to_numeric(d["tax_id"], errors="coerce")
    glen = tax.map(genome_lengths).to_numpy(dtype=float)
    reads = pd.to_numeric(d["reads"], errors="coerce").to_numpy(dtype=float)
    cov = pd.to_numeric(d["cov"], errors="coerce").to_numpy(dtype=float)

    with np.errstate(divide="ignore", invalid="ignore"):
        depth = reads * float(read_len) / glen
        denom = -np.expm1(-depth)
        ok = (np.isfinite(glen) & (glen > 0) & np.isfinite(cov) & (cov > 0)
              & np.isfinite(denom) & (denom > 1e-12))
        ev = np.where(ok, cov / np.where(denom > 1e-12, denom, np.nan), np.nan)

    # cov's denominator is the union of taxon-discriminative k-mers across every
    # strain in the database, which is not the genome. This ratio says how far
    # apart the two are; evenness_depth mixes a cov numerator with a genome
    # denominator, so it is only trustworthy where the ratio is near 1.
    # Hepatitis B virus, with ~9950 strains, sits at ~400: cov saturates at
    # 0.004 even at 12x depth, and evenness_depth wrongly calls it clumped.
    # Yersinia pestis sits at ~0.13, most of its genome being shared with
    # Y. pseudotuberculosis and so assigned above the species node.
    with np.errstate(divide="ignore", invalid="ignore"):
        kmer_set_ratio = np.where((cov > 0) & np.isfinite(glen) & (glen > 0),
                                  (d["kmers"].to_numpy(dtype=float) / cov) / glen,
                                  np.nan)

    d["genome_length"] = glen
    d["depth_estimate"] = np.where(np.isfinite(depth), depth, np.nan)
    d["evenness_depth"] = ev
    d["kmer_set_ratio"] = kmer_set_ratio
    n_ok = int(np.isfinite(ev).sum())
    print(
        f"[coverage_evenness] evenness_depth: {n_ok:,}/{n:,} taxa "
        f"(mean read length {read_len:.1f} bp)",
        file=sys.stderr, flush=True,
    )
    return d


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    Path(args.out_coverage).parent.mkdir(parents=True, exist_ok=True)

    report_dfs = []
    for path in args.reports:
        df = parse_kraken_report(path)
        if df.empty:
            print(f"[coverage_evenness] WARNING: empty report: {path}", file=sys.stderr)
        else:
            report_dfs.append(df)

    if not report_dfs:
        print(f"[coverage_evenness] WARNING: no valid reports for {args.sample_id}", file=sys.stderr)
        empty = pd.DataFrame(columns=[
            "sample_id", "tax_id", "rank", "tax_name",
            "reads", "tax_reads", "kmers", "dup", "cov", "evenness_index",
            "genome_length", "depth_estimate", "evenness_depth",
            "kmer_set_ratio",
        ])
        write_tsv(args.out_coverage, empty)
        return

    merged = merge_reports(report_dfs)

    # Filter to requested ranks
    if args.ranks:
        rank_set = set(r.lower() for r in args.ranks)
        merged = merged[merged["rank"].str.lower().isin(rank_set)].copy()

    # Filter by minimum reads
    merged = merged[merged["reads"] >= args.min_reads].copy()

    # Compute evenness
    merged = compute_evenness(merged)
    merged = compute_evenness_depth(
        merged,
        load_genome_lengths(args.species_genome_lengths),
        mean_read_length(args.unit_summaries),
    )

    # Add sample_id and reorder columns
    merged.insert(0, "sample_id", args.sample_id)

    output = merged[[
        "sample_id", "tax_id", "rank", "tax_name",
        "reads", "tax_reads", "kmers", "dup", "cov", "evenness_index",
        "genome_length", "depth_estimate", "evenness_depth", "kmer_set_ratio",
    ]].reset_index(drop=True)

    write_tsv(args.out_coverage, output)
    print(
        f"[coverage_evenness] {args.sample_id}: {len(output)} taxa written",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
