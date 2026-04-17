#!/usr/bin/env python3
"""
plot_damage_profile.py

Plot absolute-position aDNA damage profiles from adna_damage_estimate.py output.

X axis: position from read end (0 = terminal k-mer)
Y axis: fraction of unclassified k-mers (%)

One panel pair (5' end + 3' end) per taxid. Multiple samples can be overlaid
on the same axes by passing --profile and --global multiple times.

Usage:
  # Single sample
  python plot_damage_profile.py \\
    --profile sample_profile.tsv \\
    --global  sample_global.tsv \\
    --stats   sample_stats.tsv \\
    --output  damage_absolute.pdf \\
    [--taxids 1649845 632] \\
    [--max-pos 20]

  # Two samples overlaid (e.g. damaged vs undamaged)
  python plot_damage_profile.py \\
    --profile dam1_profile.tsv dam0_profile.tsv \\
    --global  dam1_global.tsv  dam0_global.tsv \\
    --stats   dam1_stats.tsv   dam0_stats.tsv \\
    --label   "aDNA damage"    "No damage" \\
    --output  damage_comparison.pdf \\
    [--taxids 1649845]

  --stats is optional but recommended: when provided, the adaptive plateau window
  and per-end p-values from adna_damage_estimate.py are used directly.
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


DEFAULT_COLORS = ["#d6604d", "#2166ac", "#4dac26", "#8073ac"]
DEFAULT_STYLES = ["-", "--", "-.", ":"]


def load_sample(profile_path, global_path, stats_path=None):
    profile = pd.read_csv(profile_path, sep="\t")
    gdf     = pd.read_csv(global_path,  sep="\t")
    sdf     = pd.read_csv(stats_path,   sep="\t") if stats_path else None
    return profile, gdf, sdf


def resolve_keys(gdfs, requested_taxids, requested_species, pvalue_threshold=0.05):
    """
    Return an ordered list of keys to plot — either int taxids or str species names.

    Explicit --taxids / --species take priority.  Otherwise auto-select all keys
    with significant evidence for aDNA damage (damage_score > 0 and
    damage_pvalue < pvalue_threshold in at least one sample), sorted by mean
    damage score descending.

    Falls back to top-6 by score when the global file predates p-value output.
    """
    if requested_species:
        return requested_species
    if requested_taxids:
        return requested_taxids

    combined = pd.concat(gdfs, ignore_index=True)
    taxid_present = (
        "taxid" in combined.columns
        and combined["taxid"].notna().any()
    )
    group_col = "taxid" if taxid_present else "species_name"

    if "damage_pvalue" in combined.columns:
        # keep keys that have at least one significant, positive-score row
        sig_mask = (
            combined["damage_score"].gt(0) &
            combined["damage_pvalue"].lt(pvalue_threshold)
        )
        sig_keys = set(combined.loc[sig_mask, group_col].dropna())
        candidates = combined[combined[group_col].isin(sig_keys)]
        keys = (
            candidates.groupby(group_col)["damage_pvalue"]
            .mean().dropna()
            .sort_values(ascending=True)
            .index.tolist()
        )
        if not keys:
            print(
                f"No taxa pass p < {pvalue_threshold} with positive damage score. "
                "Falling back to top 6 by score.",
                file=sys.stderr,
            )
        else:
            print(
                f"Auto-selected {len(keys)} taxa with damage_pvalue < {pvalue_threshold}.",
                file=sys.stderr,
            )
            return keys

    # fallback: no p-value column or nothing passed filter
    return (
        combined.groupby(group_col)["damage_score"]
        .mean().dropna()
        .sort_values(ascending=False)
        .head(3).index.tolist()
    )


def plot_end(ax, profiles, gdfs, stats_dfs, labels, colors, styles,
             key, end, max_pos, plateau_start, plateau_end):
    key_col = "species_name" if isinstance(key, str) else "taxid"

    for samp_idx, (profile, gdf, sdf, label, color, style) in enumerate(
        zip(profiles, gdfs, stats_dfs if stats_dfs else [None] * len(profiles),
            labels, colors, styles)
    ):
        sub = (profile[(profile[key_col] == key) & (profile["end"] == end)]
               .query("position < @max_pos")
               .sort_values("position"))
        if sub.empty:
            continue

        # per-sample plateau window: from stats if available, else CLI fallback
        ps, pe = plateau_start, plateau_end
        if sdf is not None:
            row = sdf[(sdf[key_col] == key) & (sdf["end"] == end)]
            if not row.empty and "plateau_pos_start" in sdf.columns:
                ps = int(row["plateau_pos_start"].iloc[0])
                pe = int(row["plateau_pos_end"].iloc[0])

        # score and p-value: prefer per-end stats, fall back to global
        pvalue = np.nan
        if sdf is not None:
            s = sdf[(sdf[key_col] == key) & (sdf["end"] == end)]
            score   = s["damage_score"].values[0]  if len(s) else np.nan
            n_reads = int(s["n_reads"].values[0])  if len(s) else 0
            if "damage_pvalue" in s.columns and len(s):
                pvalue = s["damage_pvalue"].values[0]
        else:
            g = gdf[gdf[key_col] == key]
            score_col = "damage_score" if end == "5prime" else "damage_score_3prime"
            score   = g[score_col].values[0]      if len(g) else np.nan
            n_reads = int(g["n_reads"].values[0]) if len(g) else 0

        lbl = f"{label}  (n={n_reads:,}, Δ={score*100:.2f}%"
        if not np.isnan(pvalue):
            lbl += f", p={pvalue:.1e}"
        lbl += ")"
        ax.plot(sub["position"], sub["frac_unclassified"] * 100,
                color=color, ls=style, lw=2, marker="o", markersize=3.5, label=lbl)

        # per-sample plateau: coloured vertical span + matching horizontal line
        ax.axvspan(ps, pe, alpha=0.08, color=color, lw=0)
        plateau = (sub[sub["position"].between(ps, pe)]
                   ["frac_unclassified"].mean() * 100)
        ax.axhline(plateau, color=color, lw=0.9, ls=":", alpha=0.6,
                   label=f"Plateau {ps}–{pe} ({plateau:.2f}%)")
    ax.set_title("5′ end" if end == "5prime" else "3′ end", fontsize=9)
    ax.set_xlabel("Position from read end", fontsize=8)
    ax.set_ylabel("Unclassified k-mers (%)", fontsize=8)
    ax.set_xlim(-0.5, max_pos - 0.5)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=7.5, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7)


def make_plot(profiles, gdfs, stats_dfs, labels, colors, styles,
              keys, max_pos, plateau_start, plateau_end, output_path):
    footer = (
        "Rising tail beyond plateau reflects read-length artifact "
        "(short reads: distant positions approach the opposite end)."
    )
    with PdfPages(output_path) as pdf:
        for key in keys:
            if isinstance(key, str):
                page_title = key
            else:
                name = ""
                for gdf in gdfs:
                    m = gdf[gdf["taxid"] == key]
                    if not m.empty and "species_name" in gdf.columns:
                        name = str(m["species_name"].values[0]); break
                page_title = f"taxid {key}" + (f" — {name}" if name else "")

            fig, axes = plt.subplots(1, 2, figsize=(11, 4), squeeze=False)
            fig.suptitle(
                f"aDNA damage profiles — {page_title}",
                fontsize=11,
            )
            for col, end in enumerate(["5prime", "3prime"]):
                plot_end(axes[0][col], profiles, gdfs, stats_dfs, labels, colors, styles,
                         key, end, max_pos, plateau_start, plateau_end)
            fig.text(0.5, -0.02, footer,
                     ha="center", fontsize=8, color="gray", style="italic")
            fig.tight_layout()
            pdf.savefig(fig, bbox_inches="tight", dpi=150)
            plt.close(fig)

    print(f"Saved {output_path} ({len(keys)} page(s))", file=sys.stderr)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", nargs="+", required=True,
                   help="*_profile.tsv file(s) from adna_damage_estimate.py")
    p.add_argument("--global",  nargs="+", required=True, dest="global_",
                   help="*_global.tsv file(s) from adna_damage_estimate.py")
    p.add_argument("--stats",   nargs="+", default=[],
                   help="*_stats.tsv file(s) — enables adaptive plateau shading and p-values")
    p.add_argument("--label",   nargs="+", default=[])
    p.add_argument("--output",  required=True)
    p.add_argument("--taxids",  type=int, nargs="+", default=[],
                   help="Integer taxids to plot")
    p.add_argument("--species", nargs="+", default=[],
                   help="Species names to plot (for species-aggregated profiles)")
    p.add_argument("--hits",       default=None,
                   help="all_samples.hits.tsv — use hit species for this sample as key list")
    p.add_argument("--sample-id",  default=None,
                   help="Sample ID to filter from --hits table")
    p.add_argument("--max-pos",          type=int, default=20)
    p.add_argument("--plateau-start",    type=int, default=3,
                   help="Fallback plateau start when --stats not provided (default: 3)")
    p.add_argument("--plateau-end",      type=int, default=10,
                   help="Fallback plateau end when --stats not provided (default: 10)")
    p.add_argument("--pvalue-threshold", type=float, default=0.05,
                   help="Max damage_pvalue for auto-selection of taxa (default: 0.05)")
    return p.parse_args()


def main():
    args = parse_args()
    if len(args.profile) != len(args.global_):
        print("ERROR: --profile and --global must match in count", file=sys.stderr)
        sys.exit(1)
    if args.stats and len(args.stats) != len(args.profile):
        print("ERROR: --stats count must match --profile count", file=sys.stderr)
        sys.exit(1)
    stats_paths = args.stats if args.stats else [None] * len(args.profile)
    loaded = [load_sample(p, g, s)
              for p, g, s in zip(args.profile, args.global_, stats_paths)]
    profiles  = [x[0] for x in loaded]
    gdfs      = [x[1] for x in loaded]
    stats_dfs = [x[2] for x in loaded]
    n = len(profiles)
    labels = args.label if len(args.label) == n else [f"Sample {i+1}" for i in range(n)]
    colors = [DEFAULT_COLORS[i % len(DEFAULT_COLORS)] for i in range(n)]
    styles = [DEFAULT_STYLES[i % len(DEFAULT_STYLES)] for i in range(n)]
    # --hits overrides auto-selection: use species from the hit table for this sample
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
    else:
        keys = resolve_keys(gdfs, args.taxids, args.species, args.pvalue_threshold)

    if not keys:
        print("No taxa found. Use --hits, --taxids, or --species to specify targets.",
              file=sys.stderr)
        from matplotlib.backends.backend_pdf import PdfPages
        with PdfPages(args.output) as _pdf:
            pass
        sys.exit(0)
    make_plot(profiles, gdfs, stats_dfs, labels, colors, styles,
              keys, args.max_pos, args.plateau_start, args.plateau_end, args.output)


if __name__ == "__main__":
    main()
