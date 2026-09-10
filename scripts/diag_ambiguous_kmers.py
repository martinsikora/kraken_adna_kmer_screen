#!/usr/bin/env python3
"""
diag_ambiguous_kmers.py

Diagnostic (not part of the workflow): quantify how much of the damage
profile's "unclassified" numerator is ambiguous-base k-mers ('A:' runs)
rather than genuinely unmatched k-mers ('0:' runs).

Background: parse_kmer_string_runs folds 'A:' runs into taxid 0, so the
damage estimators count ambiguous k-mers as unclassified. Ambiguous bases
cluster where base quality is worst, which is not necessarily uniform along
the read — if they concentrate at the termini they inflate the damage signal.
This script measures their positional distribution so that decision can be
made on numbers.

Mirrors screen_unit's damage gating (classified reads, read-length window)
and reports, per position from each read end:
    n_total, n_unc (taxid 0), n_ambig ('A'), frac_unc, frac_ambig,
    ambig_share = n_ambig / (n_unc + n_ambig)   <- share of the current
                                                   damage numerator that is
                                                   ambiguous k-mers
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from kraken_screen_lib import (
    open_buffered_gzip,
    parse_kmer_string_runs,
    write_tsv,
)

AMBIG = -1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Positional distribution of ambiguous vs unclassified k-mers."
    )
    parser.add_argument("--kraken-class",    required=True)
    parser.add_argument("--out",             required=True,
                        help="Per-position TSV output path")
    parser.add_argument("--max-pos",         type=int, default=10)
    parser.add_argument("--min-read-length", type=int, default=30)
    parser.add_argument("--max-read-length", type=int, default=75)
    parser.add_argument("--limit-rows",      type=int, default=0,
                        help="Stop after this many classified in-window reads "
                             "(0 = no limit)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_pos = args.max_pos
    # counts[end, pos, kind]; end 0=5', 1=3'; kind 0=total, 1=unc(0), 2=ambig(A)
    counts = np.zeros((2, max_pos, 3), dtype=np.int64)
    n_reads = 0

    with open_buffered_gzip(args.kraken_class) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t", 4)
            if len(parts) < 5 or parts[0] != "C":
                continue
            try:
                read_len = int(parts[3])
            except ValueError:
                continue
            if not (args.min_read_length <= read_len <= args.max_read_length):
                continue
            tids, cnts, _, nk, _ = parse_kmer_string_runs(
                parts[4], exclude_set=set(), ambig_taxid=AMBIG
            )
            if nk == 0:
                continue
            n_reads += 1
            kmers = np.repeat(np.array(tids, dtype=np.int64), cnts)
            n5 = min(nk, max_pos)
            for end, window in ((0, kmers[:n5]), (1, kmers[nk - n5:][::-1])):
                counts[end, :n5, 0] += 1
                counts[end, :n5, 1] += window == 0
                counts[end, :n5, 2] += window == AMBIG
            if args.limit_rows and n_reads >= args.limit_rows:
                break

    rows = []
    for end_idx, end in ((0, "5prime"), (1, "3prime")):
        for pos in range(max_pos):
            tot, unc, amb = (int(v) for v in counts[end_idx, pos])
            numer = unc + amb
            rows.append({
                "end":         end,
                "position":    pos,
                "n_total":     tot,
                "n_unc":       unc,
                "n_ambig":     amb,
                "frac_unc":    unc / tot if tot else np.nan,
                "frac_ambig":  amb / tot if tot else np.nan,
                "ambig_share": amb / numer if numer else np.nan,
            })
    df = pd.DataFrame(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    write_tsv(args.out, df)

    p0 = df[(df["end"] == "5prime") & (df["position"] == 0)].iloc[0]
    print(
        f"[diag_ambiguous_kmers] reads={n_reads:,} | 5' position 0: "
        f"frac_unc(0:)={p0['frac_unc']:.4f} frac_ambig(A:)={p0['frac_ambig']:.4f} "
        f"-> {100 * p0['ambig_share']:.2f}% of the damage numerator is ambiguous "
        f"k-mers",
        file=sys.stderr, flush=True,
    )


if __name__ == "__main__":
    main()
