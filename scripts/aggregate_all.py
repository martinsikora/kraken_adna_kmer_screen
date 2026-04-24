#!/usr/bin/env python3
"""
aggregate_all.py

Concatenate per-sample result TSVs into one integrated workflow-level summary.

Outputs:
  all_samples.summary.tsv.gz     — integrated per-sample/per-species table
                                   with stable workflow output columns
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ABUNDANCE_COLS = [
    "sample_id", "species_taxid", "species_name",
    "genus_taxid", "genus_name",
    "nnls_coefficient", "relative_abundance", "rank",
    "genus_relative_abundance", "within_genus_relative_abundance",
    "rank_within_genus",
]
DMG_COLS = [
    "sample_id", "taxid", "species_name",
    "n_reads", "damage_score", "damage_score_se",
    "damage_score_ci95_lo", "damage_score_ci95_hi",
    "damage_pvalue", "damage_score_3prime",
]
COV_COLS = [
    "sample_id", "tax_id", "tax_name", "rank",
    "reads", "kmers", "dup", "cov", "evenness_index",
]
DAMAGE_STATS_COLS = ["sample_id", "species_name", "end", "plateau_frac_unc"]
HIT_FLAG_LABELS = [
    "damage_pvalue",
    "evenness_index",
    "within_genus_relative_abundance",
    "classified_rate",
]
ALL_PASS_HIT_FLAGS = ";".join(HIT_FLAG_LABELS)
SUMMARY_OUTPUT_COLS = [
    "sample_id", "species_taxid", "species_name", "genus_taxid", "genus_name",
    "relative_abundance", "within_genus_relative_abundance", "genus_relative_abundance",
    "rank", "rank_within_genus",
    "n_reads", "damage_score", "damage_score_ci95_lo", "damage_score_ci95_hi",
    "damage_pvalue", "damage_score_3prime",
    "plateau_classified_rate",
    "evenness_index", "cov", "dup", "kmers",
    "hit_criteria_flag",
]


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
    parser.add_argument("--out-dir",     required=True,
                        help="Output directory for summary TSVs")
    parser.add_argument("--hit-max-damage-pvalue",   type=float, default=0.05,
                        help="Maximum damage_pvalue for hit table")
    parser.add_argument("--hit-min-evenness",        type=float, default=0.5,
                        help="Minimum evenness_index for hit table")
    parser.add_argument("--hit-min-within-genus-ra", type=float, default=0.1,
                        help="Minimum within_genus_relative_abundance for hit table")
    parser.add_argument("--hit-min-classified-rate", type=float, default=0.5,
                        help="Minimum plateau_classified_rate for hit table")
    return parser.parse_args()


def infer_sample_id(path: str) -> str:
    """Extract sample_id from a path like results/samples/{sample_id}/{sample_id}.*.tsv"""
    return str(Path(path).stem.split(".")[0])


def _read_tsv_subset(
    path: str,
    wanted_cols: list[str],
    nrows: int | None = None,
) -> pd.DataFrame:
    """
    Read only a requested subset of columns when possible.
    Falls back to full read+subset for edge-case parser behaviors.
    """
    wanted = set(wanted_cols)
    try:
        return pd.read_csv(
            path,
            sep="\t",
            usecols=lambda c: c in wanted,
            nrows=nrows,
        )
    except ValueError:
        # Compatibility fallback for odd files/parsers where callable usecols fails.
        df = pd.read_csv(path, sep="\t", nrows=nrows)
        present = [c for c in wanted_cols if c in df.columns]
        return df[present].copy() if present else pd.DataFrame()


def _load_typed_stack(paths: list[str], wanted_cols: list[str]) -> pd.DataFrame:
    """
    Load per-sample TSVs with narrow projected columns and concatenate.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            df = _read_tsv_subset(path, wanted_cols=wanted_cols)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if df.empty:
            continue
        if "sample_id" not in df.columns:
            df.insert(0, "sample_id", infer_sample_id(path))
        else:
            df["sample_id"] = df["sample_id"].astype(str)
        frames.append(df)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_abundance(paths: list[str]) -> pd.DataFrame:
    return _load_typed_stack(paths, ABUNDANCE_COLS)


def load_damage(paths: list[str]) -> pd.DataFrame:
    return _load_typed_stack(paths, DMG_COLS)


def load_coverage(paths: list[str]) -> pd.DataFrame:
    cov = _load_typed_stack(paths, COV_COLS)
    if cov.empty:
        return cov
    if "rank" in cov.columns:
        rank_norm = cov["rank"].astype(str).str.lower()
        cov = cov.loc[rank_norm == "species"].copy()
    return cov


def load_damage_stats_5prime(paths: list[str]) -> pd.DataFrame:
    """
    Load only the 5' damage-stat records needed for plateau_classified_rate merge.
    """
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            sdf = _read_tsv_subset(path, wanted_cols=DAMAGE_STATS_COLS)
        except (FileNotFoundError, pd.errors.EmptyDataError):
            continue
        if sdf.empty:
            continue
        if "sample_id" not in sdf.columns:
            sdf.insert(0, "sample_id", infer_sample_id(path))
        else:
            sdf["sample_id"] = sdf["sample_id"].astype(str)
        if not {"end", "species_name", "plateau_frac_unc"}.issubset(sdf.columns):
            continue
        ends = sdf["end"].astype(str).str.lower()
        sdf = sdf.loc[ends == "5prime", ["sample_id", "species_name", "plateau_frac_unc"]].copy()
        if sdf.empty:
            continue
        sdf["plateau_classified_rate"] = 1.0 - pd.to_numeric(sdf["plateau_frac_unc"], errors="coerce")
        frames.append(sdf[["sample_id", "species_name", "plateau_classified_rate"]])

    if not frames:
        return pd.DataFrame(columns=["sample_id", "species_name", "plateau_classified_rate"])
    return pd.concat(frames, ignore_index=True)


def _series_numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype=float)


def _taxid_as_string(df: pd.DataFrame, col: str) -> None:
    """
    Normalize taxid-like columns to nullable string values without '.0' artifacts.
    """
    if col not in df.columns:
        return
    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64").astype("string")


def build_integrated_summary(
    abundance_df: pd.DataFrame,
    damage_df: pd.DataFrame,
    damage_stats_5prime_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    hit_max_damage_pvalue: float,
    hit_min_evenness: float,
    hit_min_within_genus_ra: float,
    hit_min_classified_rate: float,
) -> pd.DataFrame:
    """
    Build one integrated sample-species table by outer-joining abundance, damage,
    and coverage evidence, then retain only rows with complete statistics.
    """
    abd = abundance_df[[c for c in ABUNDANCE_COLS if c in abundance_df.columns]].copy() if not abundance_df.empty else pd.DataFrame(columns=["sample_id", "species_name"])
    dmg = damage_df[[c for c in DMG_COLS if c in damage_df.columns]].copy() if not damage_df.empty else pd.DataFrame(columns=["sample_id", "species_name"])
    if "taxid" in dmg.columns:
        dmg = dmg.rename(columns={"taxid": "damage_taxid"})

    cov = pd.DataFrame(columns=["sample_id", "species_name"])
    if not coverage_df.empty:
        cov = coverage_df[[c for c in COV_COLS if c in coverage_df.columns]].copy()
        rename_map = {}
        if "tax_id" in cov.columns:
            rename_map["tax_id"] = "coverage_species_taxid"
        if "tax_name" in cov.columns:
            rename_map["tax_name"] = "species_name"
        cov = cov.rename(columns=rename_map)
        if "rank" in cov.columns:
            cov = cov.rename(columns={"rank": "coverage_rank"})

    if abd.empty and dmg.empty and cov.empty:
        return pd.DataFrame()

    merged = abd.merge(dmg, on=["sample_id", "species_name"], how="outer")
    merged = merged.merge(cov, on=["sample_id", "species_name"], how="outer")

    if "species_taxid" in merged.columns:
        merged["species_taxid"] = pd.to_numeric(merged["species_taxid"], errors="coerce")
    else:
        merged["species_taxid"] = np.nan
    if "damage_taxid" in merged.columns:
        merged["species_taxid"] = merged["species_taxid"].combine_first(
            pd.to_numeric(merged["damage_taxid"], errors="coerce")
        )
    if "coverage_species_taxid" in merged.columns:
        merged["species_taxid"] = merged["species_taxid"].combine_first(
            pd.to_numeric(merged["coverage_species_taxid"], errors="coerce")
        )
    merged["species_taxid"] = merged["species_taxid"].astype("Int64")

    if not damage_stats_5prime_df.empty:
        merged = merged.merge(
            damage_stats_5prime_df[["sample_id", "species_name", "plateau_classified_rate"]],
            on=["sample_id", "species_name"],
            how="left",
        )

    # Keep only rows where all required evidence types are present:
    # abundance + evenness + damage + classified rate.
    has_abundance = (
        _series_numeric(merged, "relative_abundance").notna()
        | _series_numeric(merged, "within_genus_relative_abundance").notna()
    )
    has_evenness = _series_numeric(merged, "evenness_index").notna()
    has_damage = _series_numeric(merged, "damage_pvalue").notna()
    has_classified_rate = _series_numeric(merged, "plateau_classified_rate").notna()
    merged = merged[has_abundance & has_evenness & has_damage & has_classified_rate].copy()

    pass_damage = (_series_numeric(merged, "damage_pvalue") < hit_max_damage_pvalue).fillna(False)
    pass_evenness = (_series_numeric(merged, "evenness_index") > hit_min_evenness).fillna(False)
    pass_within_genus = (
        _series_numeric(merged, "within_genus_relative_abundance") >= hit_min_within_genus_ra
    ).fillna(False)
    pass_classified_rate = (
        _series_numeric(merged, "plateau_classified_rate") >= hit_min_classified_rate
    ).fillna(False)

    flags = np.full(len(merged), "", dtype=object)
    for label, mask in (
        (HIT_FLAG_LABELS[0], pass_damage),
        (HIT_FLAG_LABELS[1], pass_evenness),
        (HIT_FLAG_LABELS[2], pass_within_genus),
        (HIT_FLAG_LABELS[3], pass_classified_rate),
    ):
        m = mask.to_numpy(dtype=bool, copy=False)
        flags = np.where(m, np.where(flags == "", label, flags + ";" + label), flags)
    merged["hit_criteria_flag"] = pd.Series(flags, index=merged.index, dtype="string")

    sort_cols = [c for c in ["sample_id", "species_name"] if c in merged.columns]
    if "relative_abundance" in merged.columns:
        merged["__sort_ra"] = pd.to_numeric(merged["relative_abundance"], errors="coerce")
        sort_cols = ["sample_id", "__sort_ra", "species_name"] if "sample_id" in merged.columns else ["__sort_ra", "species_name"]
        merged = merged.sort_values(sort_cols, ascending=[True, False, True], na_position="last")
        merged = merged.drop(columns=["__sort_ra"])
    elif sort_cols:
        merged = merged.sort_values(sort_cols, na_position="last")

    merged = merged.reset_index(drop=True)

    # Emit only the historical hit-table span: sample_id ... kmers.
    # Missing fields are added as NaN to keep a stable schema across runs.
    for col in SUMMARY_OUTPUT_COLS:
        if col not in merged.columns:
            merged[col] = np.nan
    _taxid_as_string(merged, "species_taxid")
    _taxid_as_string(merged, "genus_taxid")
    merged["hit_criteria_flag"] = merged["hit_criteria_flag"].astype("string").fillna("")
    return merged[SUMMARY_OUTPUT_COLS]


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

    abundance_df_all = load_abundance(args.abundance)
    damage_df_all = load_damage(args.damage)
    coverage_df = load_coverage(args.coverage)
    damage_stats_5prime_df = load_damage_stats_5prime(args.damage_stats)
    summary_df = build_integrated_summary(
        abundance_df=abundance_df_all,
        damage_df=damage_df_all,
        damage_stats_5prime_df=damage_stats_5prime_df,
        coverage_df=coverage_df,
        hit_max_damage_pvalue=args.hit_max_damage_pvalue,
        hit_min_evenness=args.hit_min_evenness,
        hit_min_within_genus_ra=args.hit_min_within_genus_ra,
        hit_min_classified_rate=args.hit_min_classified_rate,
    )
    out_path = out_dir / "all_samples.summary.tsv.gz"
    tmp_out_path = out_dir / "all_samples.summary.tsv.gz.tmp"
    summary_df.to_csv(tmp_out_path, sep="\t", index=False, compression="gzip")
    tmp_out_path.replace(out_path)

    n_hits = (
        int(summary_df["hit_criteria_flag"].fillna("").eq(ALL_PASS_HIT_FLAGS).sum())
        if not summary_df.empty
        else 0
    )
    print(
        f"[aggregate_all] rows={len(summary_df)} "
        f"samples={summary_df['sample_id'].nunique() if 'sample_id' in summary_df.columns and not summary_df.empty else 0} "
        f"hits_pass_all={n_hits} "
        f"output={out_path}",
    )


if __name__ == "__main__":
    main()
