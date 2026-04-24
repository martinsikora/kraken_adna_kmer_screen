#!/usr/bin/env python3
"""
plot_damage_summary.py

Summary biplot of aDNA damage across all taxa from workflow damage TSV outputs.

One point per taxon:
  X axis : baseline classified k-mer rate  (1 - plateau_frac_unclassified)
  Y axis : damage rate (%)                 (damage_score × 100; negative scores shown as 0)
  Size   : log10(n_reads)  — larger = more reads
  Color  : -log10(p-value) — brighter = more significant
  Shape  : △ (significant, p < threshold)  ○ (not significant)

Usage:
  python plot_damage_summary.py \\
    --stats  sample_stats.tsv \\
    --output damage_summary.pdf \\
    [--pvalue-threshold 0.05] \\
    [--min-reads 10]
"""

import argparse
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from hit_species_selection import select_hit_species


# marker size: area = (log10(n) * SIZE_SCALE) ** SIZE_EXP
SIZE_SCALE = 8.0
SIZE_EXP   = 1.5


def marker_area(log10_n: np.ndarray) -> np.ndarray:
    return (np.maximum(log10_n, 0) * SIZE_SCALE) ** SIZE_EXP


def load_stats(path: str, pvalue_threshold: float, min_reads: int) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    df = df[df["end"] == "5prime"].copy()
    df = df[df["n_reads"] >= min_reads].copy()

    df["x"] = 1.0 - df["plateau_frac_unc"]
    # clamp negative damage rates to 0
    df["y"] = df["damage_score"].clip(lower=0) * 100.0
    df["neg_log10_p"] = -np.log10(df["damage_pvalue"].clip(lower=1e-300))
    df["log10_n"] = np.log10(df["n_reads"].clip(lower=1))
    df["significant"] = (
        df["damage_pvalue"].lt(pvalue_threshold) & df["damage_score"].gt(0)
    )

    # label: prefer species_name, fall back to taxid string
    if "species_name" in df.columns:
        df["label"] = df["species_name"].fillna("").astype(str)
        taxid_mask = df["label"] == ""
        if taxid_mask.any() and "taxid" in df.columns:
            df.loc[taxid_mask, "label"] = df.loc[taxid_mask, "taxid"].astype(str)
    elif "taxid" in df.columns:
        df["label"] = df["taxid"].astype(str)
    else:
        df["label"] = ""

    return df


def make_plot(
    df: pd.DataFrame,
    pvalue_threshold: float,
    output_path: str,
    max_pos_x: float | None,
    hit_species: set[str] | None = None,
):
    vmax = max(float(df["neg_log10_p"].dropna().max()), 2.0)
    norm = mcolors.Normalize(vmin=0, vmax=vmax)
    cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(9, 6.5))

    for sig, marker in [(True, "^"), (False, "o")]:
        sub = df[df["significant"] == sig]
        if sub.empty:
            continue
        colors = cmap(norm(sub["neg_log10_p"].values))
        sizes  = marker_area(sub["log10_n"].values)
        ax.scatter(
            sub["x"], sub["y"],
            c=colors,
            s=sizes,
            marker=marker,
            edgecolors="white",
            linewidths=0.5,
            alpha=0.75,
            zorder=3,
        )

    # reference line
    ax.axhline(0, color="gray", lw=0.8, ls="--", alpha=0.5, zorder=1)

    # annotate only selected hit species from --hits filtering
    if hit_species is None:
        hit_species = set()
    to_annotate = df[df["label"].isin(hit_species)].copy()
    for _, row in to_annotate.iterrows():
        label = row["label"]
        ax.annotate(
            label,
            xy=(row["x"], row["y"]),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=6.5,
            color="#333333",
            va="bottom",
        )

    # colorbar
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.03)
    cb.set_label("-log\u2081\u2080(p-value)", fontsize=8)
    cb.ax.tick_params(labelsize=7)

    # legend — shapes + representative sizes
    legend_handles = [
        Line2D([0], [0], marker="^", color="none",
               markerfacecolor="gray", markeredgecolor="white",
               markeredgewidth=0.5, markersize=7,
               label=f"p < {pvalue_threshold} (significant)"),
        Line2D([0], [0], marker="o", color="none",
               markerfacecolor="gray", markeredgecolor="white",
               markeredgewidth=0.5, markersize=7,
               label=f"p \u2265 {pvalue_threshold}"),
    ]
    for n_ref in [100, 1_000, 10_000, 100_000]:
        s = marker_area(np.array([np.log10(n_ref)]))[0]
        legend_handles.append(
            Line2D([0], [0], marker="o", color="none",
                   markerfacecolor="gray", markeredgecolor="white",
                   markeredgewidth=0.5,
                   markersize=max(np.sqrt(s) / np.pi, 3),
                   label=f"n = {n_ref:,}")
        )
    ax.legend(handles=legend_handles, fontsize=7, frameon=False, loc="upper left")

    ax.set_xlabel("Baseline classified k-mer rate  (1 \u2212 plateau unclassified)", fontsize=9)
    ax.set_ylabel("Damage rate (%)", fontsize=9)
    ax.set_title("aDNA damage summary", fontsize=10)
    ax.set_ylim(bottom=0)
    if max_pos_x is not None:
        ax.set_xlim(right=max_pos_x)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=8)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    print(f"Saved {output_path}", file=sys.stderr)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--stats", required=True,
                   help="*_damage_stats.tsv from aggregate_sample.py")
    p.add_argument("--output", required=True,
                   help="Output plot path (.pdf or .png)")
    p.add_argument("--pvalue-threshold", type=float, default=0.05,
                   help="P-value cutoff for significant shape (default: 0.05)")
    p.add_argument("--min-reads", type=int, default=10,
                   help="Minimum reads to include a taxon (default: 10)")
    p.add_argument("--hits",      default=None,
                   help="Integrated summary table (.tsv/.tsv.gz); hit species are always annotated")
    p.add_argument("--sample-id", default=None,
                   help="Sample ID to filter from --hits table")
    p.add_argument("--hits-required-flags", nargs="+", default=[],
                   help="Required tokens in hit_criteria_flag for --hits selection "
                        "(all must be present)")
    p.add_argument("--max-keys", type=int, default=200,
                   help="Maximum number of hit species to annotate (default: 200; <=0 disables cap)")
    p.add_argument("--max-x", type=float, default=None, dest="max_pos_x",
                   help="Clip x-axis maximum (default: auto)")
    return p.parse_args()


def write_empty_plot(output_path: str, note: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")
    ax.text(
        0.5, 0.62,
        "No damage-summary points available",
        ha="center", va="center", fontsize=13, fontweight="bold",
    )
    ax.text(
        0.5, 0.42,
        note,
        ha="center", va="center", fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"Saved {output_path} (empty summary)", file=sys.stderr)


def _resolve_required_hit_flags(args: argparse.Namespace) -> list[str]:
    if args.hits_required_flags:
        return args.hits_required_flags
    return ["damage_pvalue", "within_genus_relative_abundance", "classified_rate"]


def main():
    args = parse_args()
    df = load_stats(args.stats, args.pvalue_threshold, args.min_reads)
    if df.empty:
        write_empty_plot(
            args.output,
            "No taxa passed plotting filters (end=5prime and min-reads threshold).",
        )
        sys.exit(0)
    hit_species: set[str] = set()
    required_hit_flags = _resolve_required_hit_flags(args)
    if args.hits and args.sample_id:
        try:
            selected, hit_info = select_hit_species(
                hits_path=args.hits,
                sample_id=str(args.sample_id),
                required_flag_tokens=required_hit_flags,
                max_keys=args.max_keys,
            )
            hit_species = set(selected)
            if hit_info["n_truncated"] > 0:
                print(
                    f"WARNING: selected hit species truncated to {len(hit_species)} "
                    f"(dropped {hit_info['n_truncated']} by --max-keys).",
                    file=sys.stderr,
                )
        except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError, EOFError, OSError) as e:
            print(f"WARNING: could not read --hits file ({args.hits}): {e}", file=sys.stderr)
            pass

    make_plot(
        df,
        pvalue_threshold=args.pvalue_threshold,
        output_path=args.output,
        max_pos_x=args.max_pos_x,
        hit_species=hit_species,
    )


if __name__ == "__main__":
    main()
