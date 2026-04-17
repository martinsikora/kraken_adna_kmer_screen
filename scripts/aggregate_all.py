#!/usr/bin/env python3
"""
aggregate_all.py

Concatenate per-sample result TSVs into workflow-level summary tables.

Outputs:
  all_samples.abundance.tsv      — stacked per-species NNLS results
  all_samples.damage.tsv         — stacked per-species damage scores
  all_samples.coverage.tsv       — stacked per-taxon coverage/evenness
  all_samples.summary.tsv        — one row per sample with key metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from kraken_screen_lib import write_tsv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-sample result TSVs into workflow-level summaries."
    )
    parser.add_argument("--abundance",     nargs="+", required=True,
                        help="Per-sample .abundance.tsv files")
    parser.add_argument("--damage",        nargs="+", required=True,
                        help="Per-sample .damage_global.tsv files")
    parser.add_argument("--damage-stats",  nargs="+", required=True,
                        help="Per-sample .damage_stats.tsv files")
    parser.add_argument("--coverage",      nargs="+", required=True,
                        help="Per-sample .coverage.tsv files")
    parser.add_argument("--fit",           nargs="+", required=True,
                        help="Per-sample .fit.tsv files")
    parser.add_argument("--out-dir",     required=True,
                        help="Output directory for summary TSVs")
    parser.add_argument("--min-abundance",    type=float, default=0.0001,
                        help="Minimum relative_abundance to include in all_samples.abundance.tsv")
    parser.add_argument("--min-damage-reads", type=int, default=100,
                        help="Minimum n_reads to include a taxon in all_samples.damage.tsv")
    parser.add_argument("--hit-max-damage-pvalue",   type=float, default=0.05,
                        help="Maximum damage_pvalue for hit table")
    parser.add_argument("--hit-min-evenness",        type=float, default=0.5,
                        help="Minimum evenness_index for hit table")
    parser.add_argument("--hit-min-within-genus-ra", type=float, default=0.1,
                        help="Minimum within_genus_relative_abundance for hit table")
    return parser.parse_args()


def infer_sample_id(path: str) -> str:
    """Extract sample_id from a path like results/samples/{sample_id}/{sample_id}.*.tsv"""
    return str(Path(path).stem.split(".")[0])


def load_stack(paths: list[str], min_filter_col: str | None = None, min_val: float | None = None) -> pd.DataFrame:
    """
    Load and concatenate per-sample TSVs, inferring sample_id from filename.
    Skips missing or empty files gracefully.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            df = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if df.empty:
            continue
        # Add sample_id column if not already present (coverage.tsv already has it)
        if "sample_id" not in df.columns:
            df.insert(0, "sample_id", infer_sample_id(path))
        # Apply minimum filter
        if min_filter_col and min_val is not None and min_filter_col in df.columns:
            df = df[df[min_filter_col] >= min_val].copy()
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_hit_table(
    abundance_df: pd.DataFrame,
    damage_df: pd.DataFrame,
    damage_stats_paths: list[str],
    coverage_df: pd.DataFrame,
    max_damage_pvalue: float,
    min_evenness: float,
    min_within_genus_ra: float,
) -> pd.DataFrame:
    """
    Final hit table: one row per (sample, species) passing all three criteria:
      - damage_pvalue  < max_damage_pvalue
      - evenness_index > min_evenness
      - within_genus_relative_abundance >= min_within_genus_ra

    Joins abundance (unfiltered), damage, and coverage (species rank) on
    (sample_id, species_taxid / tax_id). Missing evidence on any axis is
    treated as failing that criterion.
    """
    if abundance_df.empty or damage_df.empty or coverage_df.empty:
        return pd.DataFrame()

    # --- abundance ---
    abd = abundance_df.copy()
    if "within_genus_relative_abundance" not in abd.columns:
        # species-granularity has no within_genus column; treat RA as proxy
        abd["within_genus_relative_abundance"] = abd.get(
            "relative_abundance", pd.Series(dtype=float)
        )
    abd = abd[abd["within_genus_relative_abundance"] >= min_within_genus_ra].copy()
    if abd.empty:
        return pd.DataFrame()

    # --- damage: require significant pvalue ---
    dmg = damage_df.copy()
    if "damage_pvalue" not in dmg.columns:
        return pd.DataFrame()
    dmg = dmg[dmg["damage_pvalue"] < max_damage_pvalue].copy()
    if dmg.empty:
        return pd.DataFrame()

    # --- coverage: species rank only, require sufficient evenness ---
    cov = coverage_df[coverage_df["rank"].str.lower() == "species"].copy()
    cov = cov[cov["evenness_index"] > min_evenness].copy()
    if cov.empty:
        return pd.DataFrame()

    # --- join abundance × damage on (sample_id, species_name) ---
    dmg_cols = ["sample_id", "species_name"] + [
        c for c in ["taxid", "n_reads", "damage_score",
                    "damage_score_ci95_lo", "damage_score_ci95_hi",
                    "damage_pvalue", "damage_score_3prime"]
        if c in dmg.columns
    ]
    merged = abd.merge(dmg[dmg_cols], on=["sample_id", "species_name"], how="inner")
    if merged.empty:
        return pd.DataFrame()

    # resolve species_taxid from abundance; fall back to damage taxid column
    if "taxid" in merged.columns and "species_taxid" in merged.columns:
        merged["species_taxid"] = merged["species_taxid"].combine_first(merged.pop("taxid"))
    elif "taxid" in merged.columns:
        merged = merged.rename(columns={"taxid": "species_taxid"})

    # --- join × coverage on (sample_id, species_taxid = tax_id) ---
    cov_cols = ["sample_id", "tax_id"] + [
        c for c in ["reads", "kmers", "dup", "cov", "evenness_index"]
        if c in cov.columns
    ]
    merged = merged.merge(
        cov[cov_cols].rename(columns={"tax_id": "species_taxid"}),
        on=["sample_id", "species_taxid"],
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame()

    # --- join plateau_classified_rate from damage_stats (5prime end) ---
    stats_frames = []
    for path in damage_stats_paths:
        try:
            sdf = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if "sample_id" not in sdf.columns:
            sdf.insert(0, "sample_id", infer_sample_id(path))
        else:
            sdf["sample_id"] = sdf["sample_id"].astype(str)
        stats_frames.append(sdf)

    if stats_frames:
        stats_df = pd.concat(stats_frames, ignore_index=True)
        stats_5p = stats_df[stats_df["end"] == "5prime"][
            ["sample_id", "species_name", "plateau_frac_unc"]
        ].copy()
        stats_5p["plateau_classified_rate"] = 1.0 - stats_5p["plateau_frac_unc"]
        merged = merged.merge(
            stats_5p[["sample_id", "species_name", "plateau_classified_rate"]],
            on=["sample_id", "species_name"],
            how="left",
        )

    # --- column order ---
    lead_cols = [
        "sample_id", "species_taxid", "species_name", "genus_taxid", "genus_name",
        "relative_abundance", "within_genus_relative_abundance", "genus_relative_abundance",
        "rank", "rank_within_genus",
        "n_reads", "damage_score", "damage_score_ci95_lo", "damage_score_ci95_hi",
        "damage_pvalue", "damage_score_3prime",
        "plateau_classified_rate",
        "evenness_index", "cov", "dup", "kmers",
    ]
    ordered = [c for c in lead_cols if c in merged.columns]
    rest    = [c for c in merged.columns if c not in ordered]
    merged  = merged[ordered + rest]

    return merged.sort_values(
        ["sample_id", "relative_abundance"], ascending=[True, False]
    ).reset_index(drop=True)


def build_species_table(
    abundance_df: pd.DataFrame,
    damage_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Outer join of per-sample abundance and damage on (sample_id, species_name).

    Produces one row per (sample, species) regardless of whether the species
    has a nonzero NNLS coefficient, damage data, or both.  Missing values are
    left as NaN.
    """
    ABD_COLS = [
        "sample_id", "species_taxid", "species_name",
        "genus_taxid", "genus_name",
        "nnls_coefficient", "relative_abundance", "rank",
        "genus_relative_abundance", "within_genus_relative_abundance",
    ]
    DMG_COLS = [
        "sample_id", "taxid", "species_name",
        "n_reads", "damage_score", "damage_score_se",
        "damage_score_ci95_lo", "damage_score_ci95_hi",
        "damage_pvalue", "damage_score_3prime",
    ]

    abd = pd.DataFrame()
    if not abundance_df.empty:
        present = [c for c in ABD_COLS if c in abundance_df.columns]
        abd = abundance_df[present].copy()

    dmg = pd.DataFrame()
    if not damage_df.empty:
        present = [c for c in DMG_COLS if c in damage_df.columns]
        dmg = damage_df[present].copy()
        if "taxid" in dmg.columns and "species_taxid" not in dmg.columns:
            dmg = dmg.rename(columns={"taxid": "species_taxid"})

    if abd.empty and dmg.empty:
        return pd.DataFrame()

    if abd.empty:
        return dmg
    if dmg.empty:
        return abd

    merged = pd.merge(
        abd, dmg,
        on=["sample_id", "species_name"],
        how="outer",
        suffixes=("", "_dmg"),
    )
    # Consolidate species_taxid from both sides
    if "species_taxid_dmg" in merged.columns:
        merged["species_taxid"] = merged["species_taxid"].combine_first(
            merged.pop("species_taxid_dmg")
        )

    return merged.sort_values(
        ["sample_id", "relative_abundance"],
        ascending=[True, False],
        na_position="last",
    ).reset_index(drop=True)


def build_summary(
    abundance_df: pd.DataFrame,
    damage_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    fit_paths: list[str],
) -> pd.DataFrame:
    """
    One row per sample with key metrics drawn from all result types.
    """
    rows: list[dict] = []

    fit_lookup: dict[str, dict] = {}
    for path in fit_paths:
        try:
            df = pd.read_csv(path, sep="\t", dtype={"sample_id": str})
            if df.empty:
                continue
            sample_id = infer_sample_id(path)
            fit_lookup[sample_id] = df.iloc[0].to_dict()
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue

    all_sample_ids: set[str] = set()
    if not abundance_df.empty and "sample_id" in abundance_df.columns:
        all_sample_ids.update(abundance_df["sample_id"].unique().tolist())
    if not damage_df.empty and "sample_id" in damage_df.columns:
        all_sample_ids.update(damage_df["sample_id"].unique().tolist())
    if not coverage_df.empty and "sample_id" in coverage_df.columns:
        all_sample_ids.update(coverage_df["sample_id"].unique().tolist())
    all_sample_ids.update(fit_lookup.keys())

    for sample_id in sorted(all_sample_ids):
        row: dict = {"sample_id": sample_id}

        # From fit.tsv
        fit = fit_lookup.get(sample_id, {})
        row["n_nonzero_species"]           = fit.get("n_nonzero_species", np.nan)
        row["residual_l2_norm"]            = fit.get("residual_l2_norm", np.nan)
        row["total_explained_feature_mass"] = fit.get("total_explained_feature_mass", np.nan)
        row["fit_elapsed_seconds"]         = fit.get("fit_elapsed_seconds", np.nan)
        row["fit_granularity"]             = fit.get("fit_granularity", "")
        row["fit_mode"]                    = fit.get("fit_mode", "")

        # Top species from abundance
        if not abundance_df.empty and "sample_id" in abundance_df.columns:
            sp = abundance_df[abundance_df["sample_id"] == sample_id]
            if not sp.empty and "rank" in sp.columns:
                top = sp[sp["rank"] == 1]
                if not top.empty:
                    row["top_species_name"]       = top.iloc[0].get("species_name", "")
                    row["top_species_abundance"]  = top.iloc[0].get("relative_abundance", np.nan)
                    row["n_species_above_threshold"] = len(sp)
                else:
                    row["top_species_name"] = ""
                    row["top_species_abundance"] = np.nan
                    row["n_species_above_threshold"] = len(sp)
            else:
                row["top_species_name"] = ""
                row["top_species_abundance"] = np.nan
                row["n_species_above_threshold"] = 0

        # Top damage species
        if not damage_df.empty and "sample_id" in damage_df.columns:
            dm = damage_df[damage_df["sample_id"] == sample_id]
            if not dm.empty:
                dm_sorted = dm.sort_values("damage_score", ascending=False)
                row["top_damage_species_name"] = dm_sorted.iloc[0].get("species_name", "")
                row["top_damage_score"]        = dm_sorted.iloc[0].get("damage_score", np.nan)
                row["top_damage_pvalue"]       = dm_sorted.iloc[0].get("damage_pvalue", np.nan)
                row["n_taxa_damage_profiled"]  = len(dm)
            else:
                row["top_damage_species_name"] = ""
                row["top_damage_score"]        = np.nan
                row["top_damage_pvalue"]       = np.nan
                row["n_taxa_damage_profiled"]  = 0

        # Mean evenness from coverage
        if not coverage_df.empty and "sample_id" in coverage_df.columns:
            cov = coverage_df[
                (coverage_df["sample_id"] == sample_id)
                & (coverage_df["rank"].str.lower() == "species")
            ]
            if not cov.empty and "evenness_index" in cov.columns:
                valid = cov["evenness_index"].dropna()
                row["mean_evenness_species"] = float(valid.mean()) if len(valid) > 0 else np.nan
                row["n_species_evenness"]    = len(valid)
            else:
                row["mean_evenness_species"] = np.nan
                row["n_species_evenness"]    = 0

        rows.append(row)

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Stack per-sample TSVs
    # abundance_df: filtered to relative_abundance >= min_abundance
    # damage_df_filtered: filtered to n_reads >= min_damage_reads (for summary/damage output)
    # damage_df_all: unfiltered (for species table join — keeps all damage observations)
    abundance_df      = load_stack(args.abundance, "relative_abundance", args.min_abundance)
    damage_df_all     = load_stack(args.damage)
    damage_df         = damage_df_all[
        damage_df_all["n_reads"] >= args.min_damage_reads
    ].copy() if not damage_df_all.empty and "n_reads" in damage_df_all.columns else damage_df_all
    coverage_df       = load_stack(args.coverage)

    write_tsv(out_dir / "all_samples.abundance.tsv", abundance_df)
    write_tsv(out_dir / "all_samples.damage.tsv",    damage_df)
    write_tsv(out_dir / "all_samples.coverage.tsv",  coverage_df)

    # Per-species table: outer join of abundance + damage (all damage rows, no abundance filter)
    abundance_df_all = load_stack(args.abundance)
    species_df = build_species_table(abundance_df_all, damage_df_all)
    write_tsv(out_dir / "all_samples.species.tsv", species_df)

    # Hit table: inner join of all three sources, filtered by all three criteria
    hit_df = build_hit_table(
        abundance_df_all,
        damage_df_all,
        args.damage_stats,
        coverage_df,
        max_damage_pvalue   = args.hit_max_damage_pvalue,
        min_evenness        = args.hit_min_evenness,
        min_within_genus_ra = args.hit_min_within_genus_ra,
    )
    write_tsv(out_dir / "all_samples.hits.tsv", hit_df)

    # Summary table
    summary_df = build_summary(abundance_df, damage_df, coverage_df, args.fit)
    write_tsv(out_dir / "all_samples.summary.tsv", summary_df)

    print(
        f"[aggregate_all] samples={len(summary_df)} "
        f"abundance_rows={len(abundance_df)} "
        f"damage_rows={len(damage_df)} "
        f"species_rows={len(species_df)} "
        f"hit_rows={len(hit_df)} "
        f"coverage_rows={len(coverage_df)}",
    )


if __name__ == "__main__":
    main()
