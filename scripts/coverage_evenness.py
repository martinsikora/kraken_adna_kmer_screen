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

Output columns (all snake_case):
  sample_id, tax_id, rank, tax_name, reads, tax_reads, kmers,
  dup, cov, evenness_index
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
    return parser.parse_args()


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

    # Add sample_id and reorder columns
    merged.insert(0, "sample_id", args.sample_id)

    output = merged[[
        "sample_id", "tax_id", "rank", "tax_name",
        "reads", "tax_reads", "kmers", "dup", "cov", "evenness_index",
    ]].reset_index(drop=True)

    write_tsv(args.out_coverage, output)
    print(
        f"[coverage_evenness] {args.sample_id}: {len(output)} taxa written",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
