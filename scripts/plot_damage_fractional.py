#!/usr/bin/env python3
"""
plot_damage_fractional.py

Plot fractional-position aDNA damage profiles from pre-computed profiles
output by adna_damage_estimate.py (*_fractional_profile.tsv files).

X axis: fractional position along the read (0.0 = 5' end, 1.0 = 3' end)
Y axis: fraction of unclassified k-mers (%)

Stratifying by read length exposes:
  - Genuine terminal damage spikes at both ends (all strata)
  - The read-length artifact: for short reads, opposite-end damage zones
    overlap, producing a U-shaped profile with no flat interior
  - The plateau region is only interpretable for longer read strata

Multiple samples (e.g. damaged vs undamaged) can be overlaid per stratum
by passing multiple --fractional files.

Usage:
  # Single sample, select by taxid
  python plot_damage_fractional.py \\
    --fractional  sample_fractional_profile.tsv \\
    --taxid       1649845 \\
    --output      damage_fractional.pdf

  # Single sample, select by species name (from --species-map aggregation)
  python plot_damage_fractional.py \\
    --fractional  sample_fractional_profile.tsv \\
    --species     "Streptococcus mutans" \\
    --output      damage_fractional.pdf

  # Two samples overlaid
  python plot_damage_fractional.py \\
    --fractional  dam1_fractional_profile.tsv dam0_fractional_profile.tsv \\
    --label       "aDNA damage"  "No damage" \\
    --taxid       1649845 \\
    --output      damage_fractional_compare.pdf \\
    [--strata "31-40" "41-55" "56-75" "76-100"]
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_STRATA = [
    (31,  40,  "#d73027", "31–40 bp"),
    (41,  55,  "#fc8d59", "41–55 bp"),
    (56,  75,  "#4575b4", "56–75 bp"),
    (76, 100,  "#313695", "76–100 bp"),
]
DEFAULT_SAMPLE_COLORS = ["#d6604d", "#2166ac", "#4dac26", "#8073ac"]
DEFAULT_SAMPLE_STYLES = ["-", "--", "-.", ":"]


def resolve_keys(global_path: str | None, pvalue_threshold: float = 0.05) -> list:
    """
    Auto-select taxa to plot from a damage_global.tsv file.
    Prefers taxa with damage_pvalue < threshold, falls back to top 3 by score.
    Returns a list of species_name strings (or taxids if species_name absent).
    """
    if global_path is None:
        return []
    df = pd.read_csv(global_path, sep="\t")
    if df.empty:
        return []

    group_col = "species_name" if "species_name" in df.columns else "taxid"

    if "damage_pvalue" in df.columns:
        sig_mask = df["damage_score"].gt(0) & df["damage_pvalue"].lt(pvalue_threshold)
        sig_keys = set(df.loc[sig_mask, group_col].dropna())
        if sig_keys:
            keys = (
                df[df[group_col].isin(sig_keys)]
                .groupby(group_col)["damage_pvalue"]
                .mean().dropna()
                .sort_values(ascending=True)
                .index.tolist()
            )
            print(
                f"Auto-selected {len(keys)} taxa with damage_pvalue < {pvalue_threshold}.",
                file=sys.stderr,
            )
            return keys
        print(
            f"No taxa pass p < {pvalue_threshold}. Falling back to top 3 by score.",
            file=sys.stderr,
        )

    return (
        df.groupby(group_col)["damage_score"]
        .mean().dropna()
        .sort_values(ascending=False)
        .head(3).index.tolist()
    )


def parse_strata_spec(specs: list[str]) -> list[tuple[int, int, str, str]]:
    """Parse 'lo-hi' strings into strata tuples, reusing default colors."""
    result = []
    for i, spec in enumerate(specs):
        lo, hi = (int(x) for x in spec.split("-"))
        color = DEFAULT_STRATA[i][2] if i < len(DEFAULT_STRATA) else "#888888"
        result.append((lo, hi, color, f"{lo}–{hi} bp"))
    return result


# ---------------------------------------------------------------------------
# Load pre-computed fractional profiles
# ---------------------------------------------------------------------------

def load_fractional(path: str, taxid: int | None, species: str | None) -> pd.DataFrame:
    """
    Load a *_fractional_profile.tsv produced by adna_damage_estimate.py
    and filter to the requested taxid or species_name.
    """
    df = pd.read_csv(path, sep="\t")
    if taxid is not None:
        df = df[df["taxid"] == taxid].copy()
    elif species is not None:
        df = df[df["species_name"] == species].copy()
    return df


def build_arrays(
    df: pd.DataFrame,
    strata: list[tuple[int, int, str, str]],
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reconstruct (frac_total, frac_unc, read_counts) arrays from a filtered
    fractional profile dataframe.

    Returns:
      frac_total  shape (n_strata, n_bins)
      frac_unc    shape (n_strata, n_bins)
      read_counts shape (n_strata,)
    """
    ns = len(strata)
    frac_total  = np.zeros((ns, n_bins), dtype=np.int64)
    frac_unc    = np.zeros((ns, n_bins), dtype=np.int64)
    read_counts = np.zeros(ns, dtype=np.int64)

    for s_idx, (lo, hi, _, _) in enumerate(strata):
        sub = df[df["stratum"] == f"{lo}-{hi}"]
        if sub.empty:
            continue
        read_counts[s_idx] = int(sub["n_reads"].iloc[0])
        bins  = sub["bin"].values.astype(int)
        valid = (bins >= 0) & (bins < n_bins)
        frac_total[s_idx, bins[valid]] = sub["n_total"].values[valid].astype(np.int64)
        frac_unc[s_idx,   bins[valid]] = sub["n_unclassified"].values[valid].astype(np.int64)

    return frac_total, frac_unc, read_counts


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def make_plot(
    all_totals, all_uncs, all_read_counts,
    labels, sample_colors, sample_styles,
    strata, n_bins, key, output_path=None,
):
    n_strata = len(strata)
    bin_centers = (np.arange(n_bins) + 0.5) / n_bins

    ncols = 2
    nrows = (n_strata + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), squeeze=False)
    fig.suptitle(
        f"aDNA damage profiles — fractional position by read length\n"
        f"{key}  |  "
        + ("  ".join(f"{lbl}: solid" if i == 0 else f"{lbl}: dashed"
                     for i, lbl in enumerate(labels))
           if len(labels) > 1 else labels[0]),
        fontsize=10,
    )

    for s_idx, (lo, hi, s_color, s_label) in enumerate(strata):
        row, col = divmod(s_idx, ncols)
        ax = axes[row][col]

        for totals, uncs, read_counts, label, s_col, s_sty in zip(
            all_totals, all_uncs, all_read_counts,
            labels, sample_colors, sample_styles,
        ):
            t = totals[s_idx].astype(float)
            u = uncs[s_idx].astype(float)
            mask = t > 0
            frac = np.where(mask, u / np.where(mask, t, 1) * 100, np.nan)
            n_reads = int(read_counts[s_idx])
            lbl = f"{label}  (n={n_reads:,})"
            ax.plot(bin_centers[mask], frac[mask],
                    color=s_col, ls=s_sty, lw=2, alpha=0.9, label=lbl)

        ax.axvline(0.5, color="gray", lw=0.8, ls=":", alpha=0.4, label="Midpoint")
        ax.set_title(s_label, fontsize=9)
        ax.set_xlabel("Fractional position (0 = 5′, 1 = 3′)", fontsize=8)
        ax.set_ylabel("Unclassified k-mers (%)", fontsize=8)
        ax.set_xlim(0, 1)
        ax.set_ylim(bottom=0)
        ax.legend(fontsize=7.5, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(labelsize=7)

    for s_idx in range(n_strata, nrows * ncols):
        row, col = divmod(s_idx, ncols)
        axes[row][col].set_visible(False)

    fig.text(
        0.5, -0.01,
        "Short reads (31-40 bp): damage zones overlap across entire read, "
        "no flat interior. Longer reads show a flat central plateau.",
        ha="center", fontsize=8, color="gray", style="italic",
    )
    fig.tight_layout()
    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
        print(f"Saved {output_path}", file=sys.stderr)
    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--fractional", nargs="+", required=True,
                   help="*_fractional_profile.tsv file(s) from adna_damage_estimate.py")
    p.add_argument("--output", required=True,
                   help="Output plot path (.pdf or .png)")
    p.add_argument("--label",  nargs="+", default=[],
                   help="Sample labels (one per --fractional file)")
    p.add_argument("--strata", nargs="+", default=[],
                   help="Read-length strata to plot as 'lo-hi' (default: all four default strata)")

    key_grp = p.add_mutually_exclusive_group(required=False)
    key_grp.add_argument("--taxid",   type=int,
                         help="Integer taxid to plot (for per-taxid profiles)")
    key_grp.add_argument("--species", type=str,
                         help="Species name to plot (for species-aggregated profiles)")

    p.add_argument("--global", dest="global_tsv", default=None,
                   help="damage_global.tsv for auto-selecting significant taxa "
                        "(used when --taxid/--species are omitted and --hits not provided)")
    p.add_argument("--hits",       default=None,
                   help="all_samples.hits.tsv — use hit species for this sample as key list")
    p.add_argument("--sample-id",  default=None,
                   help="Sample ID to filter from --hits table")
    p.add_argument("--pvalue-threshold", type=float, default=0.05,
                   help="Max damage_pvalue for auto-selection (default: 0.05)")

    return p.parse_args()


def main():
    args = parse_args()

    strata = parse_strata_spec(args.strata) if args.strata else DEFAULT_STRATA

    # Resolve which taxa to plot: --hits takes priority, then explicit args, then auto
    hit_species: list[str] = []
    if args.hits and args.sample_id:
        try:
            hdf = pd.read_csv(args.hits, sep="\t", dtype={"sample_id": str})
            hit_species = (
                hdf[hdf["sample_id"] == str(args.sample_id)]["species_name"]
                .dropna().unique().tolist()
            )
        except (FileNotFoundError, pd.errors.EmptyDataError):
            pass

    if hit_species:
        keys = hit_species
    elif args.taxid is not None:
        keys = [args.taxid]
    elif args.species is not None:
        keys = [args.species]
    else:
        keys = resolve_keys(args.global_tsv, args.pvalue_threshold)

    if not keys:
        print(
            "No taxa to plot. Provide --taxid, --species, or --global with "
            "significant taxa.",
            file=sys.stderr,
        )
        # Write an empty PDF so Snakemake output is satisfied
        with PdfPages(args.output) as _pdf:
            pass
        sys.exit(0)

    n = len(args.fractional)
    labels        = args.label if len(args.label) == n else [f"Sample {i+1}" for i in range(n)]
    sample_colors = [DEFAULT_SAMPLE_COLORS[i % len(DEFAULT_SAMPLE_COLORS)] for i in range(n)]
    sample_styles = [DEFAULT_SAMPLE_STYLES[i % len(DEFAULT_SAMPLE_STYLES)] for i in range(n)]

    with PdfPages(args.output) as pdf:
        for key in keys:
            taxid   = key if isinstance(key, int) else None
            species = key if isinstance(key, str) else None

            all_dfs = [
                load_fractional(p, taxid=taxid, species=species)
                for p in args.fractional
            ]

            n_bins_vals = [int(df["bin"].max()) + 1 for df in all_dfs if not df.empty]
            if not n_bins_vals:
                print(f"WARNING: no data found for key {key!r}, skipping.", file=sys.stderr)
                continue
            n_bins = max(n_bins_vals)

            all_totals, all_uncs, all_read_counts = [], [], []
            for df in all_dfs:
                tot, unc, rc = build_arrays(df, strata, n_bins)
                all_totals.append(tot)
                all_uncs.append(unc)
                all_read_counts.append(rc)

            fig = make_plot(
                all_totals, all_uncs, all_read_counts,
                labels, sample_colors, sample_styles,
                strata, n_bins, key, output_path=None,
            )
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {args.output} ({len(keys)} page(s))", file=sys.stderr)


if __name__ == "__main__":
    main()
