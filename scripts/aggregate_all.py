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
    "reads", "tax_reads", "kmers", "dup", "cov", "evenness_index",
    # Carried for the leads table only; the summary does not use them.
    # kmer_set_ratio is the reference k-mer set size over a single genome's
    # worth, so it says how far `cov` is from "fraction of one genome
    # covered": >>1 deflates cov (HBV sits near 400 with ~10k database
    # sequences), <<1 inflates it (a partial reference).
    "genome_length", "depth_estimate", "evenness_depth", "kmer_set_ratio",
]
DAMAGE_STATS_COLS = ["sample_id", "taxid", "species_name", "end", "plateau_frac_unc"]
HIT_FLAG_LABELS = [
    # damage significance AND a biologically possible damage rate; see
    # _pass_damage_criterion
    "damage_rate",
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
    # Per-base damage rates from the k-mer window deconvolution
    # (damage_model.tsv), placed next to the plateau statistics: the plateau
    # damage_score/damage_pvalue are the detection statistic (conservative
    # lower bound), the model amplitude damage_rate_5prime is the effect-size
    # estimate, robust to the fragment-length distribution.
    # terminal rate = interior_rate + damage_rate_*, so it is not carried here.
    # Only damage_rate_5prime feeds a hit criterion (as an implausibility veto).
    "interior_rate", "damage_rate_5prime", "damage_rate_5prime_se",
    "damage_rate_3prime", "damage_model_pvalue_5prime",
    "plateau_classified_rate",
    "evenness_index", "cov", "dup", "kmers",
    "hit_criteria_flag",
]

DAMAGE_MODEL_COLS = [
    "sample_id", "taxid", "species_name",
    "interior_rate", "damage_rate_5prime", "damage_rate_5prime_se",
    "damage_rate_3prime", "damage_model_pvalue_5prime",
]


def _taxid_int64(df: pd.DataFrame, col: str) -> None:
    """Normalize a taxid column to nullable Int64 in place (before merging)."""
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


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
    parser.add_argument("--damage-model", nargs="*", default=None,
                        help="Per-sample .damage_model.tsv files. Adds the "
                             "per-base damage rates to the summary as "
                             "annotations; omitted columns are written as NaN")
    parser.add_argument("--out-dir",     required=True,
                        help="Output directory for summary TSVs")
    parser.add_argument("--abundance-only-min-within-genus-ra", type=float, default=0.1,
                        help="Keep abundance-only rows (below evenness_min_reads, so "
                             "absent from coverage.tsv) at or above this within-genus "
                             "relative abundance. Default 0.1")
    parser.add_argument("--out-file",    default=None,
                        help="Write the summary to this exact path instead of "
                             "<out-dir>/all_samples.summary.tsv.gz. Every column "
                             "of the summary, hit_criteria_flag included, is a "
                             "per-row function of one sample's own evidence, so "
                             "running this script on a single sample yields "
                             "exactly that sample's slice of the full table; "
                             "this option is what lets the workflow build a "
                             "per-sample summary without waiting for the rest.")
    parser.add_argument("--hit-max-damage-pvalue",   type=float, default=0.05,
                        help="Maximum damage_pvalue for the damage_rate criterion")
    parser.add_argument("--hit-max-damage-rate",     type=float, default=0.4,
                        help="Maximum damage_rate_5prime for the damage_rate "
                             "criterion. Single-strand deamination cannot exceed "
                             "~0.5 per base, so a fitted rate above this is not a "
                             "damage profile; taxa without a model fit are not "
                             "rejected by it")
    parser.add_argument("--hit-min-evenness",        type=float, default=0.5,
                        help="Minimum evenness_index for hit table (legacy mode, "
                             "and fallback when dup/cov are unavailable)")
    parser.add_argument("--hit-min-within-genus-ra", type=float, default=0.1,
                        help="Minimum within_genus_relative_abundance for hit table")
    parser.add_argument("--hit-min-classified-rate", type=float, default=0.5,
                        help="Minimum plateau_classified_rate for hit table")
    parser.add_argument("--hit-evenness-mode", choices=["depth-aware", "legacy"],
                        default="depth-aware",
                        help="How to evaluate the evenness_index hit criterion. "
                             "depth-aware applies a duplication test to shallow taxa "
                             "and a breadth test to deep ones; legacy thresholds the "
                             "raw evenness_index (pre-existing behaviour)")
    parser.add_argument("--hit-evenness-lambda-split", type=float, default=0.1,
                        help="Mean genome depth (dup*cov) separating the shallow and "
                             "deep regimes in depth-aware mode")
    parser.add_argument("--hit-max-dup-shallow",     type=float, default=6.0,
                        help="Shallow regime: maximum dup (k-mer duplication)")
    parser.add_argument("--hit-min-cov-deep",        type=float, default=0.02,
                        help="Deep regime: minimum cov (breadth of k-mer coverage)")
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

    sample_id is forced to str: an all-digit id such as 018345 would otherwise
    be typed as int64 and lose its leading zero, so the sample_id carried in
    coverage.tsv would no longer match the one inferred from the file path and
    every row of that sample would drop out of the merge.
    """
    wanted = set(wanted_cols)
    try:
        return pd.read_csv(
            path,
            sep="\t",
            dtype={"sample_id": str},
            usecols=lambda c: c in wanted,
            nrows=nrows,
        )
    except ValueError:
        # Compatibility fallback for odd files/parsers where callable usecols fails.
        df = pd.read_csv(path, sep="\t", dtype={"sample_id": str}, nrows=nrows)
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
        if "taxid" not in sdf.columns:
            continue
        ends = sdf["end"].astype(str).str.lower()
        sdf = sdf.loc[ends == "5prime", ["sample_id", "taxid", "plateau_frac_unc"]].copy()
        if sdf.empty:
            continue
        sdf["plateau_classified_rate"] = 1.0 - pd.to_numeric(sdf["plateau_frac_unc"], errors="coerce")
        frames.append(sdf[["sample_id", "taxid", "plateau_classified_rate"]])

    if not frames:
        return pd.DataFrame(columns=["sample_id", "taxid", "plateau_classified_rate"])
    out = pd.concat(frames, ignore_index=True)
    _taxid_int64(out, "taxid")
    return out


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


def _pass_evenness_criterion(
    merged: pd.DataFrame,
    mode: str,
    hit_min_evenness: float,
    lambda_split: float,
    max_dup_shallow: float,
    min_cov_deep: float,
) -> pd.Series:
    """
    Evaluate the evenness_index hit criterion.

    evenness_index is the Lander-Waterman ratio E = cov / (1 - exp(-dup*cov)),
    which degenerates at both ends of the depth range:

        dup*cov << 1  ->  1 - exp(-lambda) ~= lambda,  so E ~= 1/dup
        dup*cov >> 1  ->  1 - exp(-lambda) ~= 1,       so E ~= cov

    Real aDNA screening data sits almost entirely in the first regime (in the
    reference dataset, 82% of rows have lambda < 0.01, where E matches 1/dup to
    within 0.1%). A single threshold on E therefore means "dup < 1/threshold"
    for shallow taxa but "at least this fraction of the genome covered" for deep
    ones -- two unrelated tests. In practice that rejected a 354k-read, p=0
    Yersinia pestis hit with 35% breadth while accepting a 372-read one.

    depth-aware mode applies the test appropriate to each regime instead:
        shallow (lambda <  lambda_split): dup < max_dup_shallow
        deep    (lambda >= lambda_split): cov > min_cov_deep

    legacy mode thresholds E directly, reproducing the previous behaviour.
    Rows without usable dup/cov fall back to the legacy test in either mode.
    """
    evenness = _series_numeric(merged, "evenness_index")
    legacy = (evenness > hit_min_evenness).fillna(False)
    if mode == "legacy":
        return legacy

    dup = _series_numeric(merged, "dup")
    cov = _series_numeric(merged, "cov")
    lam = dup * cov
    usable = dup.notna() & cov.notna() & lam.notna()

    depth_aware = pd.Series(
        np.where(lam < lambda_split, dup < max_dup_shallow, cov > min_cov_deep),
        index=merged.index,
    ).fillna(False)

    return depth_aware.where(usable, legacy).astype(bool)


def load_damage_model(paths: list | None) -> pd.DataFrame:
    """Stack per-sample damage_model.tsv files, keeping only the rate columns."""
    if not paths:
        return pd.DataFrame(columns=DAMAGE_MODEL_COLS)
    df = _load_typed_stack(list(paths), wanted_cols=DAMAGE_MODEL_COLS)
    if df.empty or "taxid" not in df.columns:
        return pd.DataFrame(columns=DAMAGE_MODEL_COLS)
    _taxid_int64(df, "taxid")
    # one row per (sample, taxon); the fit is already per taxon
    return df.drop_duplicates(subset=["sample_id", "taxid"])


def build_integrated_summary(
    abundance_df: pd.DataFrame,
    damage_df: pd.DataFrame,
    damage_stats_5prime_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    damage_model_df: pd.DataFrame,
    hit_max_damage_pvalue: float,
    hit_max_damage_rate: float,
    hit_min_evenness: float,
    hit_min_within_genus_ra: float,
    hit_min_classified_rate: float,
    hit_evenness_mode: str = "depth-aware",
    hit_evenness_lambda_split: float = 0.1,
    hit_max_dup_shallow: float = 6.0,
    hit_min_cov_deep: float = 0.02,
) -> pd.DataFrame:
    """
    Build one integrated sample-species table by outer-joining abundance, damage,
    and coverage evidence, then retain only rows with complete statistics.
    """
    # All merges key on (sample_id, species_taxid). Names differ between the
    # KrakenUniq report and the taxonomy dump (which would silently drop rows
    # from a name join); taxids do not. species_name is coalesced afterwards.
    abd = abundance_df[[c for c in ABUNDANCE_COLS if c in abundance_df.columns]].copy() if not abundance_df.empty else pd.DataFrame(columns=["sample_id", "species_taxid", "species_name"])
    _taxid_int64(abd, "species_taxid")

    dmg = damage_df[[c for c in DMG_COLS if c in damage_df.columns]].copy() if not damage_df.empty else pd.DataFrame(columns=["sample_id", "taxid", "species_name"])
    dmg = dmg.rename(columns={"taxid": "species_taxid",
                              "species_name": "damage_species_name"})
    _taxid_int64(dmg, "species_taxid")

    cov = pd.DataFrame(columns=["sample_id", "species_taxid"])
    if not coverage_df.empty:
        cov = coverage_df[[c for c in COV_COLS if c in coverage_df.columns]].copy()
        cov = cov.rename(columns={"tax_id":   "species_taxid",
                                  "tax_name": "coverage_species_name",
                                  "rank":     "coverage_rank"})
        _taxid_int64(cov, "species_taxid")

    if abd.empty and dmg.empty and cov.empty:
        return pd.DataFrame()

    merged = abd.merge(dmg, on=["sample_id", "species_taxid"], how="outer")
    merged = merged.merge(cov, on=["sample_id", "species_taxid"], how="outer")

    # species_name: abundance (taxonomy dump) -> damage -> KrakenUniq report
    name = merged["species_name"] if "species_name" in merged.columns else pd.Series(pd.NA, index=merged.index)
    name = name.replace("", pd.NA)
    for fallback in ["damage_species_name", "coverage_species_name"]:
        if fallback in merged.columns:
            name = name.combine_first(merged[fallback].replace("", pd.NA))
    merged["species_name"] = name
    merged = merged.drop(columns=[c for c in ["damage_species_name", "coverage_species_name"] if c in merged.columns])

    if not damage_stats_5prime_df.empty:
        stats = damage_stats_5prime_df.rename(columns={"taxid": "species_taxid"})
        merged = merged.merge(
            stats[["sample_id", "species_taxid", "plateau_classified_rate"]],
            on=["sample_id", "species_taxid"],
            how="left",
        )

    # Per-base damage rates, left-joined so a taxon without a model fit keeps
    # its row: the fit needs read-length strata above a minimum, so it covers
    # fewer taxa than the damage profile does.
    if damage_model_df is not None and not damage_model_df.empty:
        model = damage_model_df.rename(columns={"taxid": "species_taxid"})
        cols = [c for c in ["sample_id", "species_taxid", "interior_rate",
                            "damage_rate_5prime", "damage_rate_5prime_se",
                            "damage_rate_3prime", "damage_model_pvalue_5prime"]
                if c in model.columns]
        merged = merged.merge(model[cols],
                              on=["sample_id", "species_taxid"], how="left")

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

    # damage_rate: significant damage AND a rate that damage could produce.
    # Deamination in a single-stranded overhang saturates around 0.5 per base,
    # so a fitted terminal rate above hit_max_damage_rate is not damage but a
    # taxon whose reads mismatch the reference throughout -- the profile's
    # terminal excess is then an artefact of misassignment. Measured here:
    # Hydrogenimonas cancrithermarum 0.59 and Arcobacter venerupis 0.34, both
    # at ~160 reads with near-zero interior_rate, so an interior-rate test would
    # not catch them. A taxon with no model fit has NaN and is left to the
    # p-value alone rather than rejected, since absence of a fit is not evidence.
    rate = _series_numeric(merged, "damage_rate_5prime")
    implausible = (rate > hit_max_damage_rate).fillna(False)
    pass_damage = (
        (_series_numeric(merged, "damage_pvalue") < hit_max_damage_pvalue).fillna(False)
        & ~implausible
    )
    pass_evenness = _pass_evenness_criterion(
        merged,
        mode=hit_evenness_mode,
        hit_min_evenness=hit_min_evenness,
        lambda_split=hit_evenness_lambda_split,
        max_dup_shallow=hit_max_dup_shallow,
        min_cov_deep=hit_min_cov_deep,
    )
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


LEADS_EXTRA_COLS = [
    "evidence", "reads", "tax_reads", "genome_length", "depth_estimate",
    "evenness_depth", "kmer_set_ratio",
]


def build_leads_table(
    summary_df: pd.DataFrame,
    abundance_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    min_within_genus_ra: float,
) -> pd.DataFrame:
    """Superset of the integrated summary, with damage left-joined.

    The summary requires abundance, evenness, damage and classified-rate
    statistics all to be present, so a taxon below damage_min_reads (counted
    *after* the damage read-length window) vanishes from it entirely even when
    its coverage and abundance are informative. On a 742-sample screen that
    hid, among others, four Hepatitis B samples covering roughly a third of the
    genome at a duplication rate near 2, and four of the seven Yersinia pestis
    samples that a targeted mapping workflow recovered.

    Rows are labelled by what supports them, and the summary is exactly the
    `evidence == "full"` subset:

      full           abundance + coverage + damage (a detection)
      coverage_only  no damage profile: too few reads in the length window
      abundance_only below evenness_min_reads, so not even in coverage.tsv;
                     kept only above min_within_genus_ra, since without reads
                     or k-mers the within-genus share is the only signal

    Rows that are not `full` carry NO damage evidence. They are leads for
    targeted follow-up, never authenticated detections, and must not be
    counted alongside hits.
    """
    cov = coverage_df.rename(columns={"tax_id": "species_taxid", "tax_name": "species_name"}).copy()
    cov = cov.drop(columns=[c for c in ("rank",) if c in cov.columns])
    abu_keys = [c for c in ABUNDANCE_COLS if c not in ("species_name",)]
    abu_all = abundance_df.copy()
    # build_integrated_summary normalizes taxids to strings, so every frame that
    # takes part in these joins has to agree on that: merging int64 against
    # string keys is a hard error in pandas, not a silent miss.
    for frame in (cov, abu_all):
        _taxid_as_string(frame, "species_taxid")
        frame["sample_id"] = frame["sample_id"].astype(str)
    abu = abu_all[[c for c in abu_keys if c in abu_all.columns]]

    base = cov.merge(abu, on=["sample_id", "species_taxid"], how="left")

    # Abundance-only rows: present in the fit but below evenness_min_reads.
    seen = set(zip(cov["sample_id"], cov["species_taxid"]))
    abu_only = abu_all[
        [(s, t) not in seen for s, t in zip(abu_all["sample_id"], abu_all["species_taxid"])]
    ]
    if "within_genus_relative_abundance" in abu_only.columns:
        abu_only = abu_only[
            pd.to_numeric(abu_only["within_genus_relative_abundance"], errors="coerce")
            >= min_within_genus_ra
        ]
    leads = pd.concat([base, abu_only], ignore_index=True)

    dmg_cols = [c for c in SUMMARY_OUTPUT_COLS if c not in cov.columns and c not in abu.columns]
    dmg = summary_df[["sample_id", "species_taxid"] + [c for c in dmg_cols if c in summary_df.columns]].copy()
    _taxid_as_string(dmg, "species_taxid")
    dmg["sample_id"] = dmg["sample_id"].astype(str)
    leads["sample_id"] = leads["sample_id"].astype(str)
    _taxid_as_string(leads, "species_taxid")
    leads = leads.merge(dmg, on=["sample_id", "species_taxid"], how="left", suffixes=("", "_dup"))
    leads = leads.drop(columns=[c for c in leads.columns if c.endswith("_dup")])

    has_damage = leads["damage_pvalue"].notna() if "damage_pvalue" in leads.columns else False
    has_cov = leads["kmers"].notna() if "kmers" in leads.columns else False
    leads["evidence"] = np.where(has_damage, "full", np.where(has_cov, "coverage_only", "abundance_only"))

    # Concatenating the abundance-only rows promotes any int column that gains a
    # NaN to float, which would write genus_taxid as "6.0" and break string joins
    # on it. Normalize the taxid columns the way the summary does.
    for col in ("species_taxid", "genus_taxid"):
        _taxid_as_string(leads, col)

    ordered = [c for c in SUMMARY_OUTPUT_COLS if c in leads.columns]
    ordered = ordered[:3] + ["evidence"] + [c for c in ordered[3:]]
    ordered += [c for c in LEADS_EXTRA_COLS if c in leads.columns and c not in ordered]
    return leads[[c for c in ordered if c in leads.columns]]


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    abundance_df_all = load_abundance(args.abundance)
    damage_df_all = load_damage(args.damage)
    coverage_df = load_coverage(args.coverage)
    damage_stats_5prime_df = load_damage_stats_5prime(args.damage_stats)
    damage_model_df = load_damage_model(args.damage_model)
    summary_df = build_integrated_summary(
        abundance_df=abundance_df_all,
        damage_df=damage_df_all,
        damage_stats_5prime_df=damage_stats_5prime_df,
        coverage_df=coverage_df,
        damage_model_df=damage_model_df,
        hit_max_damage_pvalue=args.hit_max_damage_pvalue,
        hit_max_damage_rate=args.hit_max_damage_rate,
        hit_min_evenness=args.hit_min_evenness,
        hit_min_within_genus_ra=args.hit_min_within_genus_ra,
        hit_min_classified_rate=args.hit_min_classified_rate,
        hit_evenness_mode=args.hit_evenness_mode,
        hit_evenness_lambda_split=args.hit_evenness_lambda_split,
        hit_max_dup_shallow=args.hit_max_dup_shallow,
        hit_min_cov_deep=args.hit_min_cov_deep,
    )
    # The written table is the superset: `summary_df` is exactly its
    # `evidence == "full"` subset, so nothing is lost by not writing it too.
    out_df = build_leads_table(
        summary_df=summary_df,
        abundance_df=abundance_df_all,
        coverage_df=coverage_df,
        min_within_genus_ra=args.abundance_only_min_within_genus_ra,
    )
    out_path = Path(args.out_file) if args.out_file else out_dir / "all_samples.summary.tsv.gz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out_path = out_path.with_suffix(out_path.suffix + ".tmp")
    out_df.to_csv(tmp_out_path, sep="\t", index=False, compression="gzip")
    tmp_out_path.replace(out_path)

    n_hits = (
        int(summary_df["hit_criteria_flag"].fillna("").eq(ALL_PASS_HIT_FLAGS).sum())
        if not summary_df.empty
        else 0
    )
    tiers = out_df["evidence"].value_counts().to_dict()
    print(
        f"[aggregate_all] rows={len(out_df)} "
        f"full={tiers.get('full', 0)} "
        f"coverage_only={tiers.get('coverage_only', 0)} "
        f"abundance_only={tiers.get('abundance_only', 0)} "
        f"samples={summary_df['sample_id'].nunique() if 'sample_id' in summary_df.columns and not summary_df.empty else 0} "
        f"hits_pass_all={n_hits} "
        f"output={out_path}",
    )


if __name__ == "__main__":
    main()
