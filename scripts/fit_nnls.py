#!/usr/bin/env python3
"""
fit_nnls.py

Fit species abundances with NNLS from a sparse reference matrix and a
sparse observed sample vector.

Adapted from prototype 1 (kraken_kmer_species_abundance/scripts/fit_nnls.py).
Changes from prototype:
  - Import from kraken_screen_lib instead of kraken_abundance
  - Use load_genus_membership_v2 (new column schema)
  - Always write .genus.tsv (empty header-only TSV when granularity=species)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize, nnls

from kraken_screen_lib import (
    load_sparse_matrix,
    load_sparse_matrix_for_column_slicing,
    load_sparse_vector,
    load_genus_membership_v2,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit species abundances with NNLS from a sparse reference matrix and sparse observed vector."
    )
    parser.add_argument("--reference-matrix",   required=True)
    parser.add_argument("--species-metadata",   required=True)
    parser.add_argument("--sample-vector",      required=True)
    parser.add_argument("--reference-features", default="",
                        help="Feature index TSV; required for --restrict-to-target-genus-features.")
    parser.add_argument("--out-prefix",         required=True)
    parser.add_argument("--fit-mode",
                        choices=("exact", "fast"), default="exact")
    parser.add_argument("--fit-constraint",
                        choices=("nnls", "simplex"), default="nnls")
    parser.add_argument("--fit-granularity",
                        choices=("species", "genus", "within-genus"), default="species")
    parser.add_argument("--max-candidates",         type=int,   default=512)
    parser.add_argument("--max-feature-support",    type=int,   default=0)
    parser.add_argument("--genus-taxids",           default="")
    parser.add_argument("--target-genus",           action="append", default=[])
    parser.add_argument("--target-genus-file",      action="append", default=[])
    parser.add_argument("--restrict-to-target-genus-features", action="store_true")
    parser.add_argument("--min-genus-relative-abundance", type=float, default=0.0)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_sample_stats(sample_vector: str | Path) -> Dict[str, object]:
    summary_path = Path(str(sample_vector).replace(".vector.npz", ".summary.tsv"))
    if summary_path.exists():
        return pd.read_csv(summary_path, sep="\t").iloc[0].to_dict()
    return {}


def load_target_genus_specs(target_genera: list[str], target_genus_files: list[str]) -> list[str]:
    raw: list[str] = [s.strip() for s in target_genera if s.strip()]
    for path in target_genus_files:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            content = line.split("#", 1)[0].strip()
            if content:
                raw.extend(t for t in content.replace(",", " ").split() if t.strip())
    seen: set[str] = set()
    deduped: list[str] = []
    for s in raw:
        if s not in seen:
            deduped.append(s)
            seen.add(s)
    return deduped


def load_species_and_genus_metadata(
    species_metadata: str | Path,
    genus_taxids_path: str | Path,
) -> tuple[pd.DataFrame, Dict[int, Tuple[int, str]]]:
    species_df = pd.read_csv(species_metadata, sep="\t")
    species_df = species_df.sort_values("species_index").reset_index(drop=True)
    gpath = Path(genus_taxids_path)
    genus_lookup = load_genus_membership_v2(gpath) if gpath.exists() else {}
    species_name_lookup = species_df.set_index("species_taxid")["species_name"].to_dict()
    species_df["genus_taxid"] = species_df["species_taxid"].map(
        lambda t: int(genus_lookup[t][0]) if t in genus_lookup else int(t)
    )
    species_df["genus_name"] = species_df["species_taxid"].map(
        lambda t: str(genus_lookup[t][1]) if t in genus_lookup else str(species_name_lookup[int(t)])
    )
    return species_df, genus_lookup


def load_reference_feature_taxids(reference_features: str | Path) -> np.ndarray:
    df = pd.read_csv(reference_features, sep="\t").sort_values("feature_index").reset_index(drop=True)
    expected = np.arange(len(df), dtype=np.int64)
    if not np.array_equal(df["feature_index"].to_numpy(dtype=np.int64), expected):
        raise ValueError(f"{reference_features}: feature_index not contiguous from 0")
    return df["taxid"].to_numpy(dtype=np.int64, copy=False)


def filter_sample_features_to_target_genera(
    sample_feature_list: np.ndarray,
    sample_data: np.ndarray,
    reference_feature_taxids: np.ndarray,
    genus_lookup: Dict[int, Tuple[int, str]],
    target_genus_taxids: set[int],
) -> tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    if sample_feature_list.size == 0:
        return sample_feature_list, sample_data, {
            "restrict_to_target_genus_features": True,
            "n_active_features_before_target_genus_feature_filter": 0,
            "n_active_features_after_target_genus_feature_filter": 0,
            "active_feature_mass_before_target_genus_feature_filter": 0.0,
            "active_feature_mass_after_target_genus_feature_filter": 0.0,
            "excluded_target_genus_feature_mass": 0.0,
            "target_genus_feature_taxids": "",
        }
    observed_feature_taxids = reference_feature_taxids[sample_feature_list]
    mapped_genus_taxids = np.fromiter(
        (int(genus_lookup.get(int(t), (int(t), ""))[0]) for t in observed_feature_taxids),
        dtype=np.int64,
        count=int(observed_feature_taxids.size),
    )
    keep_mask = np.isin(mapped_genus_taxids, np.asarray(sorted(target_genus_taxids), dtype=np.int64))
    filtered_fl   = sample_feature_list[keep_mask]
    filtered_data = sample_data[keep_mask]
    filtered_mass = float(filtered_data.sum())
    feature_taxids = observed_feature_taxids[keep_mask]
    return filtered_fl, filtered_data, {
        "restrict_to_target_genus_features": True,
        "n_active_features_before_target_genus_feature_filter": int(sample_feature_list.size),
        "n_active_features_after_target_genus_feature_filter": int(filtered_fl.size),
        "active_feature_mass_before_target_genus_feature_filter": float(sample_data.sum()),
        "active_feature_mass_after_target_genus_feature_filter": filtered_mass,
        "excluded_target_genus_feature_mass": float(sample_data.sum() - filtered_mass),
        "target_genus_feature_taxids": ";".join(
            str(t) for t in sorted(set(int(x) for x in feature_taxids.tolist()))
        ),
    }


def resolve_target_genera(
    species_df: pd.DataFrame,
    target_genus_specs: list[str],
) -> tuple[pd.DataFrame, np.ndarray, Dict[str, object]]:
    if not target_genus_specs:
        return species_df.copy(), np.arange(len(species_df), dtype=np.int32), {
            "target_genus_count_requested": 0,
            "target_genus_count_matched": 0,
            "target_genus_count_fitted": 0,
            "target_genus_names": "",
            "target_genus_taxids": "",
        }
    matched_taxids: set[int] = set()
    matched_names:  set[str] = set()
    for spec in target_genus_specs:
        token = str(spec).strip()
        if not token:
            continue
        if token.lstrip("-").isdigit():
            mask = species_df["genus_taxid"] == int(token)
        else:
            mask = species_df["genus_name"] == token
        if not mask.any():
            continue
        matched = species_df.loc[mask, ["genus_taxid", "genus_name"]].drop_duplicates()
        matched_taxids.update(int(v) for v in matched["genus_taxid"].tolist())
        matched_names.update(str(v)  for v in matched["genus_name"].tolist())
    if not matched_taxids:
        raise ValueError("No target genera matched: " + ", ".join(target_genus_specs))
    selected_mask    = species_df["genus_taxid"].isin(matched_taxids).to_numpy()
    selected_indices = np.flatnonzero(selected_mask).astype(np.int32, copy=False)
    filtered_df      = species_df.loc[selected_mask].reset_index(drop=True)
    return filtered_df, selected_indices, {
        "target_genus_count_requested": int(len(target_genus_specs)),
        "target_genus_count_matched":   int(len(matched_taxids)),
        "target_genus_count_fitted":    int(len(matched_taxids)),
        "target_genus_names":  ";".join(sorted(matched_names)),
        "target_genus_taxids": ";".join(str(t) for t in sorted(matched_taxids)),
    }


def build_genus_metadata(
    species_df: pd.DataFrame,
    species_reference: sparse.csr_matrix,
) -> tuple[pd.DataFrame, sparse.csr_matrix, list[np.ndarray]]:
    genus_rows: list[dict] = []
    genus_species_indices: list[np.ndarray] = []
    species_to_genus = np.empty(len(species_df), dtype=np.int32)
    grouped = species_df.groupby(["genus_taxid", "genus_name"], sort=False, dropna=False)
    for genus_index, ((genus_taxid, genus_name), group) in enumerate(grouped):
        species_indices = group.index.to_numpy(dtype=np.int32, copy=False)
        genus_species_indices.append(species_indices)
        species_to_genus[species_indices] = genus_index
        genus_rows.append({
            "genus_index":           int(genus_index),
            "genus_taxid":           int(genus_taxid),
            "genus_name":            str(genus_name),
            "n_species":             int(len(species_indices)),
            "total_reference_weight": float(group["total_reference_weight"].sum()),
        })
    coo = species_reference.tocoo()
    genus_row_indices = species_to_genus[coo.row]
    genus_matrix = sparse.coo_matrix(
        (coo.data, (genus_row_indices, coo.col)),
        shape=(len(genus_rows), species_reference.shape[1]),
    ).tocsr()
    return pd.DataFrame(genus_rows), genus_matrix, genus_species_indices


def build_within_genus_groups(
    species_df: pd.DataFrame,
) -> list[tuple[int, str, np.ndarray]]:
    groups: list[tuple[int, str, np.ndarray]] = []
    for _, ((genus_taxid, genus_name), group) in enumerate(
        species_df.groupby(["genus_taxid", "genus_name"], sort=False, dropna=False)
    ):
        groups.append((int(genus_taxid), str(genus_name), group.index.to_numpy(dtype=np.int32, copy=False)))
    return groups


def prepare_active_feature_subset(
    reference_matrix: sparse.spmatrix,
    sample_feature_list: np.ndarray,
    sample_data: np.ndarray,
    max_feature_support: int,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, Dict[str, object], np.ndarray]:
    stage_start = time.perf_counter()
    active_mass_before = float(sample_data.sum())
    reference_subset = reference_matrix[:, sample_feature_list]
    cand_mask_before = np.asarray(reference_subset.getnnz(axis=1)).ravel() > 0
    cand_before      = int(cand_mask_before.sum())

    if max_feature_support > 0 and reference_subset.shape[1] > 0:
        feat_support = np.asarray(reference_subset.getnnz(axis=0)).ravel()
        keep = feat_support <= max_feature_support
        sample_feature_list = sample_feature_list[keep]
        sample_data         = sample_data[keep]
        reference_subset    = reference_subset[:, keep]

    reference_subset = reference_subset.tocsr()
    cand_mask  = np.asarray(reference_subset.getnnz(axis=1)).ravel() > 0
    cand_list  = np.flatnonzero(cand_mask)
    fit_info   = {
        "candidate_species_before_filter": cand_before,
        "candidate_species_after_filter":  int(cand_list.size),
        "n_active_features_before_filter": int(cand_mask_before.size),
        "n_active_features_after_filter":  int(sample_feature_list.size),
        "active_feature_mass_before_filter": active_mass_before,
        "active_feature_mass_after_filter":  float(sample_data.sum()),
        "fit_elapsed_seconds": float(time.perf_counter() - stage_start),
    }
    return reference_subset, sample_feature_list, sample_data, fit_info, cand_list


def solve_simplex_constrained(
    candidate_reference: sparse.csr_matrix,
    sample_data: np.ndarray,
) -> tuple[np.ndarray, float, bool, str]:
    a = candidate_reference.transpose().toarray().astype(np.float64, copy=False)
    n = a.shape[1]
    if n == 0:
        return np.zeros(0, dtype=np.float64), 0.0, True, "empty"
    if n == 1:
        coef = np.array([1.0], dtype=np.float64)
        return coef, float(np.linalg.norm(a[:, 0] - sample_data)), True, "single-candidate"
    ata = a.T @ a
    atb = a.T @ sample_data
    snorm_sq = float(np.dot(sample_data, sample_data))
    positive = np.clip(atb, 0.0, None)
    ps = float(positive.sum())
    x0 = positive / ps if ps > 0 else np.full(n, 1.0 / n, dtype=np.float64)

    result = minimize(
        lambda x: float(0.5 * (x @ ata @ x - 2.0 * x @ atb + snorm_sq)),
        x0,
        method="SLSQP",
        jac=lambda x: ata @ x - atb,
        bounds=[(0.0, 1.0)] * n,
        constraints=({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0),
                      "jac": lambda x: np.ones_like(x, dtype=np.float64)},),
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    coef = np.clip(result.x.astype(np.float64), 0.0, 1.0)
    cs = float(coef.sum())
    coef = coef / cs if cs > 0 else np.full(n, 1.0 / n, dtype=np.float64)
    return coef, float(np.linalg.norm(a @ coef - sample_data)), bool(result.success), str(result.message)


def fit_candidate_rows(
    reference_subset: sparse.csr_matrix,
    candidate_list: np.ndarray,
    sample_data: np.ndarray,
    fit_mode: str,
    fit_constraint: str,
    max_candidates: int,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, float, int, Dict[str, object]]:
    cand_after = int(candidate_list.size)
    if candidate_list.size == 0:
        empty_ref = reference_subset[candidate_list, :].tocsr()
        return candidate_list, empty_ref, np.zeros(0, dtype=np.float64), 0.0, cand_after, {
            "solver_success": True, "solver_status": "empty", "solver_message": "empty",
        }
    if fit_mode == "fast" and candidate_list.size > max_candidates:
        scores = np.asarray(reference_subset[candidate_list, :].dot(sample_data)).ravel()
        pos_mask = scores > 0
        if not np.all(pos_mask):
            candidate_list = candidate_list[pos_mask]
            scores = scores[pos_mask]
        if candidate_list.size == 0:
            empty_ref = reference_subset[candidate_list, :].tocsr()
            return candidate_list, empty_ref, np.zeros(0, dtype=np.float64), 0.0, 0, {
                "solver_success": True, "solver_status": "empty", "solver_message": "empty",
            }
        if candidate_list.size > max_candidates:
            pos = np.argpartition(scores, -max_candidates)[-max_candidates:]
            ord_ = pos[np.argsort(scores[pos])[::-1]]
            candidate_list = candidate_list[ord_]
        cand_after = int(candidate_list.size)
    if candidate_list.size == 0:
        empty_ref = reference_subset[candidate_list, :].tocsr()
        return candidate_list, empty_ref, np.zeros(0, dtype=np.float64), 0.0, cand_after, {
            "solver_success": True, "solver_status": "empty", "solver_message": "empty",
        }
    candidate_reference = reference_subset[candidate_list, :].tocsr()
    if fit_constraint == "simplex":
        coef, res_norm, ok, msg = solve_simplex_constrained(candidate_reference, sample_data)
        solver_info = {"solver_success": bool(ok), "solver_status": "success" if ok else "fallback", "solver_message": msg}
    else:
        a = candidate_reference.transpose().toarray()
        coef, res_norm = nnls(a, sample_data)
        solver_info = {"solver_success": True, "solver_status": "success", "solver_message": "nnls"}
    return candidate_list, candidate_reference, coef, float(res_norm), cand_after, solver_info


def format_result_frame(
    row_df: pd.DataFrame,
    candidate_list: np.ndarray,
    coefficients: np.ndarray,
    candidate_reference: sparse.csr_matrix,
    fit_constraint: str,
) -> tuple[pd.DataFrame, float, float, int]:
    coef_sum = float(coefficients.sum())
    explained_mass = np.asarray(candidate_reference.sum(axis=1)).ravel() * coefficients
    explained_sum  = float(explained_mass.sum())
    if fit_constraint == "simplex":
        abund_values = coefficients
        abund_sum    = coef_sum
    else:
        abund_values = explained_mass
        abund_sum    = explained_sum
    result_rows: List[Dict] = []
    for pos, coef in enumerate(coefficients):
        if coef <= 0:
            continue
        row = row_df.iloc[int(candidate_list[pos])].to_dict()
        row.update({
            "nnls_coefficient":            float(coef),
            "coefficient_relative_abundance": float(coef / coef_sum) if coef_sum > 0 else 0.0,
            "explained_feature_mass":      float(explained_mass[pos]),
            "relative_abundance":          float(abund_values[pos] / abund_sum) if abund_sum > 0 else 0.0,
        })
        result_rows.append(row)
    result_df = pd.DataFrame(result_rows)
    if not result_df.empty:
        result_df = result_df.sort_values(
            ["relative_abundance", "explained_feature_mass", "nnls_coefficient"],
            ascending=False,
        ).reset_index(drop=True)
        result_df["rank"] = np.arange(1, len(result_df) + 1, dtype=np.int32)
    else:
        result_df["rank"] = pd.Series(dtype=np.int32)
    return result_df, coef_sum, explained_sum, int((coefficients > 0).sum())


def fit_level(
    reference_matrix: sparse.spmatrix,
    row_df: pd.DataFrame,
    sample_feature_list: np.ndarray,
    sample_data: np.ndarray,
    fit_mode: str,
    fit_constraint: str,
    max_candidates: int,
    max_feature_support: int,
) -> tuple[pd.DataFrame, Dict[str, object], np.ndarray, np.ndarray]:
    stage_start = time.perf_counter()
    reference_subset, filtered_features, filtered_data, subset_info, candidate_list = \
        prepare_active_feature_subset(reference_matrix, sample_feature_list, sample_data, max_feature_support)

    result_columns = list(row_df.columns) + [
        "nnls_coefficient", "coefficient_relative_abundance",
        "explained_feature_mass", "relative_abundance", "rank",
    ]
    if candidate_list.size == 0:
        empty = row_df.iloc[0:0].copy()
        for col in ["nnls_coefficient", "coefficient_relative_abundance",
                    "explained_feature_mass", "relative_abundance", "rank"]:
            if col not in empty.columns:
                empty[col] = pd.Series(dtype=np.float64 if col != "rank" else np.int32)
        empty = empty.reindex(columns=result_columns, fill_value=np.nan)
        fit_info = {
            **subset_info,
            "n_candidate_species": 0, "n_nonzero_species": 0,
            "residual_l2_norm": 0.0, "total_explained_feature_mass": 0.0,
            "candidate_species_after_shortlist": 0,
            "fit_mode": fit_mode, "fit_constraint": fit_constraint,
            "max_candidates": int(max_candidates), "max_feature_support": int(max_feature_support),
            "stage_elapsed_seconds": float(time.perf_counter() - stage_start),
        }
        return empty, fit_info, filtered_features, filtered_data

    candidate_list, candidate_reference, coef, res_norm, cand_after, solver_info = fit_candidate_rows(
        reference_subset, candidate_list, filtered_data, fit_mode, fit_constraint, max_candidates,
    )
    result_df, coef_sum, explained_sum, n_nonzero = format_result_frame(
        row_df, candidate_list, coef, candidate_reference, fit_constraint,
    )
    result_df = result_df.reindex(columns=result_columns, fill_value=np.nan)
    fit_info = {
        **subset_info,
        "n_candidate_species": int(candidate_list.size),
        "n_nonzero_species": int(n_nonzero),
        "residual_l2_norm": float(res_norm if filtered_data.size else 0.0),
        "total_explained_feature_mass": float(explained_sum),
        "candidate_species_after_shortlist": int(cand_after),
        "fit_mode": fit_mode, "fit_constraint": fit_constraint,
        "solver_success": bool(solver_info.get("solver_success", True)),
        "solver_status": str(solver_info.get("solver_status", "")),
        "solver_message": str(solver_info.get("solver_message", "")),
        "max_candidates": int(max_candidates), "max_feature_support": int(max_feature_support),
        "stage_elapsed_seconds": float(time.perf_counter() - stage_start),
    }
    return result_df, fit_info, filtered_features, filtered_data


def empty_output_frame(template: pd.DataFrame) -> pd.DataFrame:
    empty = template.iloc[0:0].copy()
    for col in ["nnls_coefficient", "coefficient_relative_abundance",
                "explained_feature_mass", "relative_abundance", "rank"]:
        if col not in empty.columns:
            empty[col] = pd.Series(dtype=np.float64 if col != "rank" else np.int32)
    return empty


def add_species_within_genus_metrics(result_df: pd.DataFrame) -> pd.DataFrame:
    if result_df.empty:
        out = result_df.copy()
        for col in ["genus_relative_abundance", "within_genus_relative_abundance",
                    "within_genus_nnls_coefficient"]:
            out[col] = pd.Series(dtype=np.float64)
        out["rank_within_genus"] = pd.Series(dtype=np.int32)
        return out
    out = result_df.copy()
    genus_abund = out.groupby("genus_taxid", sort=False)["relative_abundance"].transform("sum")
    genus_coef  = out.groupby("genus_taxid", sort=False)["nnls_coefficient"].transform("sum")
    out["genus_relative_abundance"]       = genus_abund
    out["within_genus_relative_abundance"] = np.where(
        genus_abund.to_numpy(dtype=np.float64) > 0.0,
        out["relative_abundance"].to_numpy(dtype=np.float64) / genus_abund.to_numpy(dtype=np.float64),
        0.0,
    )
    out["within_genus_nnls_coefficient"] = np.where(
        genus_coef.to_numpy(dtype=np.float64) > 0.0,
        out["nnls_coefficient"].to_numpy(dtype=np.float64) / genus_coef.to_numpy(dtype=np.float64),
        0.0,
    )
    out = out.sort_values(
        ["genus_taxid", "within_genus_relative_abundance", "relative_abundance", "species_name"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    out["rank_within_genus"] = out.groupby("genus_taxid", sort=False).cumcount() + 1
    out = out.sort_values(
        ["relative_abundance", "explained_feature_mass", "nnls_coefficient"],
        ascending=False,
    ).reset_index(drop=True)
    out["rank"] = np.arange(1, len(out) + 1, dtype=np.int32)
    out["rank_within_genus"] = out["rank_within_genus"].astype(np.int32, copy=False)
    return out


# ---------------------------------------------------------------------------
# Empty genus TSV schema (written when granularity=species)
# ---------------------------------------------------------------------------

GENUS_TSV_COLUMNS = [
    "genus_index", "genus_taxid", "genus_name", "n_species", "total_reference_weight",
]


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)

    sample_vector      = load_sparse_vector(args.sample_vector)
    sample_summary     = load_sample_stats(args.sample_vector)
    sample_coo         = sample_vector.tocoo()
    sample_feature_list = sample_coo.col.astype(np.int32, copy=False)
    sample_data        = sample_coo.data.astype(np.float64, copy=False)
    n_features_before  = int(sample_feature_list.size)
    mass_before        = float(sample_data.sum())
    target_genus_specs = load_target_genus_specs(args.target_genus, args.target_genus_file)

    if args.fit_granularity == "species":
        species_df, genus_lookup = load_species_and_genus_metadata(
            args.species_metadata, args.genus_taxids,
        )
        species_df = species_df.sort_values("species_index").reset_index(drop=True)
        reference_matrix = load_sparse_matrix_for_column_slicing(args.reference_matrix)
        reference_matrix_csr = reference_matrix.tocsr()
        species_df, target_indices, target_info = resolve_target_genera(species_df, target_genus_specs)
        if len(target_indices) != reference_matrix_csr.shape[0]:
            reference_matrix_csr = reference_matrix_csr[target_indices, :]
            reference_matrix     = reference_matrix[target_indices, :]

        target_ff_info: Dict[str, object] = {
            "restrict_to_target_genus_features": False,
            "n_active_features_before_target_genus_feature_filter": n_features_before,
            "n_active_features_after_target_genus_feature_filter":  n_features_before,
            "active_feature_mass_before_target_genus_feature_filter": mass_before,
            "active_feature_mass_after_target_genus_feature_filter":  mass_before,
            "excluded_target_genus_feature_mass": 0.0,
            "target_genus_feature_taxids": "",
        }
        if args.restrict_to_target_genus_features:
            if not target_genus_specs:
                raise ValueError("--restrict-to-target-genus-features requires --target-genus")
            if not args.reference_features:
                raise ValueError("--restrict-to-target-genus-features requires --reference-features")
            ref_feat_taxids = load_reference_feature_taxids(args.reference_features)
            tg_taxids = set(int(t) for t in species_df["genus_taxid"].dropna().astype(int).unique().tolist())
            sample_feature_list, sample_data, target_ff_info = filter_sample_features_to_target_genera(
                sample_feature_list, sample_data, ref_feat_taxids, genus_lookup, tg_taxids,
            )

        fit_df, fit_info, _, _ = fit_level(
            reference_matrix, species_df, sample_feature_list, sample_data,
            args.fit_mode, args.fit_constraint, args.max_candidates, args.max_feature_support,
        )
        fit_df = add_species_within_genus_metrics(fit_df)
        fit_info["fit_elapsed_seconds"] = float(time.perf_counter() - start_time)
        fit_info["n_active_features_before_any_filter"]   = n_features_before
        fit_info["active_feature_mass_before_any_filter"] = mass_before
        fit_info["fit_granularity"] = args.fit_granularity
        fit_info.update(target_ff_info)
        write_tsv(out_prefix.with_suffix(".abundance.tsv"), fit_df)
        write_tsv(out_prefix.with_suffix(".fit.tsv"), pd.DataFrame([fit_info | target_info | sample_summary]))
        # Always write genus.tsv (empty for species granularity)
        write_tsv(out_prefix.with_suffix(".genus.tsv"), pd.DataFrame(columns=GENUS_TSV_COLUMNS))
        return

    # genus / within-genus granularity
    species_df, genus_lookup = load_species_and_genus_metadata(args.species_metadata, args.genus_taxids)
    species_df = species_df.sort_values("species_index").reset_index(drop=True)
    species_reference     = load_sparse_matrix_for_column_slicing(args.reference_matrix)
    species_reference_csr = species_reference.tocsr()
    species_df, target_indices, target_info = resolve_target_genera(species_df, target_genus_specs)
    if len(target_indices) != species_reference_csr.shape[0]:
        species_reference_csr = species_reference_csr[target_indices, :]
        species_reference     = species_reference[target_indices, :]

    target_ff_info = {
        "restrict_to_target_genus_features": False,
        "n_active_features_before_target_genus_feature_filter": n_features_before,
        "n_active_features_after_target_genus_feature_filter":  n_features_before,
        "active_feature_mass_before_target_genus_feature_filter": mass_before,
        "active_feature_mass_after_target_genus_feature_filter":  mass_before,
        "excluded_target_genus_feature_mass": 0.0,
        "target_genus_feature_taxids": "",
    }
    if args.restrict_to_target_genus_features:
        if not target_genus_specs:
            raise ValueError("--restrict-to-target-genus-features requires --target-genus")
        if not args.reference_features:
            raise ValueError("--restrict-to-target-genus-features requires --reference-features")
        ref_feat_taxids = load_reference_feature_taxids(args.reference_features)
        tg_taxids = set(int(t) for t in species_df["genus_taxid"].dropna().astype(int).unique().tolist())
        sample_feature_list, sample_data, target_ff_info = filter_sample_features_to_target_genera(
            sample_feature_list, sample_data, ref_feat_taxids, genus_lookup, tg_taxids,
        )

    n_feat_before_filter = int(sample_feature_list.size)
    mass_before_filter   = float(sample_data.sum())

    if n_feat_before_filter == 0:
        empty_fit = {
            "fit_granularity": args.fit_granularity,
            "n_candidate_species": 0, "n_nonzero_species": 0,
            "residual_l2_norm": 0.0, "total_explained_feature_mass": 0.0,
            "max_feature_support": int(args.max_feature_support),
            "fit_mode": args.fit_mode, "fit_constraint": args.fit_constraint,
            "max_candidates": int(args.max_candidates),
            "n_active_features_before_filter": 0,
            "n_active_features_after_filter": 0,
            "active_feature_mass_before_filter": 0.0,
            "active_feature_mass_after_filter": 0.0,
            "candidate_species_before_filter": 0,
            "candidate_species_after_filter": 0,
            "candidate_species_after_shortlist": 0,
            "fit_elapsed_seconds": float(time.perf_counter() - start_time),
            **target_ff_info, **target_info, **sample_summary,
        }
        write_tsv(out_prefix.with_suffix(".abundance.tsv"), pd.DataFrame())
        write_tsv(out_prefix.with_suffix(".genus.tsv"),     pd.DataFrame())
        write_tsv(out_prefix.with_suffix(".fit.tsv"),       pd.DataFrame([empty_fit]))
        return

    if args.fit_granularity == "within-genus":
        _, filtered_features, filtered_data, global_filter_info, _ = prepare_active_feature_subset(
            species_reference, sample_feature_list, sample_data, args.max_feature_support,
        )
        species_result_frames: list[pd.DataFrame] = []
        genus_summary_rows: list[dict] = []
        genus_fit_rows: list[dict] = []
        sp_cand_before_total = sp_cand_after_total = sp_cand_shortlist_total = 0
        sp_nonzero_total = 0
        sp_elapsed_total = 0.0
        total_explained = 0.0
        total_residual  = 0.0

        for genus_taxid, genus_name, sp_indices in build_within_genus_groups(species_df):
            sp_subset_df     = species_df.iloc[sp_indices].reset_index(drop=True)
            sp_subset_matrix = species_reference_csr[sp_indices, :]
            sp_fit_df, sp_fit_info, _, _ = fit_level(
                sp_subset_matrix, sp_subset_df, filtered_features, filtered_data,
                args.fit_mode, args.fit_constraint, args.max_candidates, 0,
            )
            sp_elapsed_total     += float(sp_fit_info["stage_elapsed_seconds"])
            sp_cand_before_total += int(sp_fit_info["candidate_species_before_filter"])
            sp_cand_after_total  += int(sp_fit_info["candidate_species_after_filter"])
            sp_cand_shortlist_total += int(sp_fit_info["candidate_species_after_shortlist"])
            sp_nonzero_total     += int(sp_fit_info["n_nonzero_species"])
            total_explained      += float(sp_fit_info["total_explained_feature_mass"])
            total_residual       += float(sp_fit_info["residual_l2_norm"])

            best_sp_name, best_sp_abund, best_sp_rank = "", 0.0, 0
            if not sp_fit_df.empty:
                best = sp_fit_df.iloc[0]
                best_sp_name  = str(best["species_name"])
                best_sp_abund = float(best["relative_abundance"])
                best_sp_rank  = int(best["rank"])
                sp_fit_df["genus_taxid"] = int(genus_taxid)
                sp_fit_df["genus_name"]  = str(genus_name)
                sp_fit_df["within_genus_relative_abundance"] = sp_fit_df["relative_abundance"]
                sp_fit_df["within_genus_nnls_coefficient"]   = sp_fit_df["nnls_coefficient"]
                sp_fit_df["rank_within_genus"] = sp_fit_df["rank"]
                species_result_frames.append(sp_fit_df)
            genus_summary_rows.append({
                "genus_taxid": int(genus_taxid), "genus_name": str(genus_name),
                "n_species": int(len(sp_indices)),
                "candidate_species_before_filter":   int(sp_fit_info["candidate_species_before_filter"]),
                "candidate_species_after_filter":    int(sp_fit_info["candidate_species_after_filter"]),
                "candidate_species_after_shortlist": int(sp_fit_info["candidate_species_after_shortlist"]),
                "n_nonzero_species":                 int(sp_fit_info["n_nonzero_species"]),
                "residual_l2_norm":                  float(sp_fit_info["residual_l2_norm"]),
                "total_explained_feature_mass":      float(sp_fit_info["total_explained_feature_mass"]),
                "fit_elapsed_seconds":               float(sp_fit_info["stage_elapsed_seconds"]),
                "best_species_name":                 best_sp_name,
                "best_species_relative_abundance":   best_sp_abund,
                "best_species_rank_within_genus":    best_sp_rank,
            })
            genus_fit_rows.append({
                "genus_taxid": int(genus_taxid), "genus_name": str(genus_name),
                "genus_explained_feature_mass": float(sp_fit_info["total_explained_feature_mass"]),
                "genus_residual_l2_norm":        float(sp_fit_info["residual_l2_norm"]),
            })

        species_result_df = (
            pd.concat(species_result_frames, ignore_index=True)
            if species_result_frames
            else empty_output_frame(species_df)
        )
        genus_summary_df = pd.DataFrame(genus_summary_rows)
        if not genus_summary_df.empty:
            genus_fit_df = pd.DataFrame(genus_fit_rows)
            tot_genus_mass = float(genus_fit_df["genus_explained_feature_mass"].sum())
            if tot_genus_mass > 0:
                genus_summary_df = genus_summary_df.merge(genus_fit_df, on=["genus_taxid", "genus_name"], how="left")
                genus_summary_df["genus_relative_abundance"] = genus_summary_df["genus_explained_feature_mass"] / tot_genus_mass
            else:
                genus_summary_df["genus_explained_feature_mass"] = 0.0
                genus_summary_df["genus_residual_l2_norm"]       = 0.0
                genus_summary_df["genus_relative_abundance"]     = 0.0
            genus_summary_df = genus_summary_df.sort_values(["genus_name", "genus_taxid"]).reset_index(drop=True)
            keep_genus_mask  = genus_summary_df["genus_relative_abundance"] > float(args.min_genus_relative_abundance)
            keep_genus_tids  = set(genus_summary_df.loc[keep_genus_mask, "genus_taxid"].astype(int).tolist())
            if not species_result_df.empty:
                species_result_df = species_result_df[species_result_df["genus_taxid"].isin(keep_genus_tids)].copy()
                if not species_result_df.empty:
                    species_result_df = species_result_df.merge(
                        genus_summary_df[["genus_taxid", "genus_relative_abundance"]],
                        on="genus_taxid", how="left",
                    ).sort_values(
                        ["relative_abundance", "explained_feature_mass", "nnls_coefficient"],
                        ascending=False,
                    ).reset_index(drop=True)
                    species_result_df["rank"] = np.arange(
                        1, len(species_result_df) + 1, dtype=np.int32
                    )

        write_tsv(out_prefix.with_suffix(".abundance.tsv"), species_result_df)
        write_tsv(out_prefix.with_suffix(".genus.tsv"),     genus_summary_df)
        fit_summary = {
            "fit_granularity": args.fit_granularity, "fit_mode": args.fit_mode,
            "fit_constraint": args.fit_constraint, "max_candidates": int(args.max_candidates),
            "max_feature_support": int(args.max_feature_support),
            "n_active_features_before_filter": n_feat_before_filter,
            "n_active_features_after_filter":  int(len(filtered_features)),
            "active_feature_mass_before_filter": mass_before_filter,
            "active_feature_mass_after_filter":  float(filtered_data.sum()),
            "candidate_species_before_filter":   int(global_filter_info["candidate_species_before_filter"]),
            "candidate_species_after_filter":    int(global_filter_info["candidate_species_after_filter"]),
            "candidate_species_after_shortlist": int(global_filter_info["candidate_species_after_filter"]),
            "n_candidate_species":    int(global_filter_info["candidate_species_after_filter"]),
            "n_nonzero_species":      int(sp_nonzero_total),
            "residual_l2_norm":       float(total_residual),
            "total_explained_feature_mass": float(total_explained),
            "genus_candidate_species_before_filter_total":   int(sp_cand_before_total),
            "genus_candidate_species_after_filter_total":    int(sp_cand_after_total),
            "genus_candidate_species_after_shortlist_total": int(sp_cand_shortlist_total),
            "genus_n_nonzero_total":       int(sp_nonzero_total),
            "genus_fit_elapsed_seconds_total": float(sp_elapsed_total),
            "n_genus_fitted":   int(len(genus_summary_df)),
            "n_genus_retained": int(
                (genus_summary_df["genus_relative_abundance"] > float(args.min_genus_relative_abundance)).sum()
            ) if not genus_summary_df.empty else 0,
            "min_genus_relative_abundance": float(args.min_genus_relative_abundance),
            "fit_elapsed_seconds": float(time.perf_counter() - start_time),
            **target_ff_info, **target_info, **sample_summary,
        }
        write_tsv(out_prefix.with_suffix(".fit.tsv"), pd.DataFrame([fit_summary]))
        return

    # genus granularity
    genus_df, genus_reference, genus_species_indices = build_genus_metadata(species_df, species_reference_csr)
    genus_fit_df, genus_fit_info, filtered_features, filtered_data = fit_level(
        genus_reference, genus_df, sample_feature_list, sample_data,
        args.fit_mode, args.fit_constraint, args.max_candidates, args.max_feature_support,
    )
    species_result_frames_g: list[pd.DataFrame] = []
    sp_cand_before_total_g = sp_cand_after_total_g = sp_cand_shortlist_total_g = 0
    sp_nonzero_total_g = 0
    sp_elapsed_total_g = 0.0

    for genus_result in genus_fit_df.itertuples(index=False):
        if genus_result.relative_abundance <= 0:
            continue
        genus_index = int(genus_result.genus_index)
        sp_indices  = genus_species_indices[genus_index]
        sp_subset_df     = species_df.iloc[sp_indices].reset_index(drop=True)
        sp_subset_matrix = species_reference_csr[sp_indices, :]
        sp_fit_df, sp_fit_info, _, _ = fit_level(
            sp_subset_matrix, sp_subset_df, filtered_features, filtered_data,
            args.fit_mode, args.fit_constraint, args.max_candidates, args.max_feature_support,
        )
        sp_elapsed_total_g      += float(sp_fit_info["stage_elapsed_seconds"])
        sp_cand_before_total_g  += int(sp_fit_info["candidate_species_before_filter"])
        sp_cand_after_total_g   += int(sp_fit_info["candidate_species_after_filter"])
        sp_cand_shortlist_total_g += int(sp_fit_info["candidate_species_after_shortlist"])
        sp_nonzero_total_g      += int(sp_fit_info["n_nonzero_species"])
        if sp_fit_df.empty:
            continue
        sp_fit_df["genus_index"]               = genus_index
        sp_fit_df["genus_taxid"]               = int(genus_result.genus_taxid)
        sp_fit_df["genus_name"]                = str(genus_result.genus_name)
        sp_fit_df["genus_nnls_coefficient"]    = float(genus_result.nnls_coefficient)
        sp_fit_df["genus_relative_abundance"]  = float(genus_result.relative_abundance)
        sp_fit_df["genus_explained_feature_mass"] = float(genus_result.explained_feature_mass)
        sp_fit_df["within_genus_relative_abundance"] = sp_fit_df["relative_abundance"]
        sp_fit_df["within_genus_nnls_coefficient"]   = sp_fit_df["nnls_coefficient"]
        sp_fit_df["rank_within_genus"] = sp_fit_df["rank"]
        sp_fit_df["relative_abundance"] = (
            sp_fit_df["genus_relative_abundance"] * sp_fit_df["within_genus_relative_abundance"]
        )
        sp_fit_df["explained_feature_mass"] = (
            sp_fit_df["genus_explained_feature_mass"] * sp_fit_df["within_genus_relative_abundance"]
        )
        species_result_frames_g.append(sp_fit_df)

    if species_result_frames_g:
        species_result_df_g = pd.concat(species_result_frames_g, ignore_index=True)
        species_result_df_g = species_result_df_g.sort_values(
            ["relative_abundance", "explained_feature_mass", "nnls_coefficient"],
            ascending=False,
        ).reset_index(drop=True)
        species_result_df_g["rank"] = np.arange(1, len(species_result_df_g) + 1, dtype=np.int32)
    else:
        species_result_df_g = empty_output_frame(species_df)

    genus_summary_df_g = genus_fit_df.copy()
    if not genus_summary_df_g.empty:
        genus_summary_df_g = genus_summary_df_g.sort_values(
            ["genus_name", "genus_taxid"]
        ).reset_index(drop=True)

    write_tsv(out_prefix.with_suffix(".abundance.tsv"), species_result_df_g)
    write_tsv(out_prefix.with_suffix(".genus.tsv"),     genus_summary_df_g)
    fit_summary_g = {
        "fit_granularity": args.fit_granularity, "fit_mode": args.fit_mode,
        "fit_constraint": args.fit_constraint, "max_candidates": int(args.max_candidates),
        "max_feature_support": int(args.max_feature_support),
        "n_active_features_before_filter": n_feat_before_filter,
        "n_active_features_after_filter":  int(len(filtered_features)),
        "active_feature_mass_before_filter": mass_before_filter,
        "active_feature_mass_after_filter":  float(filtered_data.sum()),
        "candidate_species_before_filter":   int(genus_fit_info["candidate_species_before_filter"]),
        "candidate_species_after_filter":    int(genus_fit_info["candidate_species_after_filter"]),
        "candidate_species_after_shortlist": int(genus_fit_info["candidate_species_after_shortlist"]),
        "n_candidate_species":   int(genus_fit_info["n_candidate_species"]),
        "n_nonzero_species":     int(genus_fit_info["n_nonzero_species"]),
        "residual_l2_norm":      float(genus_fit_info["residual_l2_norm"]),
        "total_explained_feature_mass": float(genus_fit_info["total_explained_feature_mass"]),
        "genus_candidate_species_before_filter": int(genus_fit_info["candidate_species_before_filter"]),
        "genus_candidate_species_after_filter":  int(genus_fit_info["candidate_species_after_filter"]),
        "genus_candidate_species_after_shortlist": int(genus_fit_info["candidate_species_after_shortlist"]),
        "genus_n_nonzero":                 int(genus_fit_info["n_nonzero_species"]),
        "genus_residual_l2_norm":          float(genus_fit_info["residual_l2_norm"]),
        "genus_total_explained_feature_mass": float(genus_fit_info["total_explained_feature_mass"]),
        "genus_fit_elapsed_seconds":       float(genus_fit_info["stage_elapsed_seconds"]),
        "species_candidate_species_before_filter_total":   int(sp_cand_before_total_g),
        "species_candidate_species_after_filter_total":    int(sp_cand_after_total_g),
        "species_candidate_species_after_shortlist_total": int(sp_cand_shortlist_total_g),
        "species_n_nonzero_total":   int(sp_nonzero_total_g),
        "species_fit_elapsed_seconds": float(sp_elapsed_total_g),
        "n_genus_fitted": int((genus_fit_df["relative_abundance"] > 0).sum()) if not genus_fit_df.empty else 0,
        "fit_elapsed_seconds": float(time.perf_counter() - start_time),
        **target_ff_info, **target_info, **sample_summary,
    }
    write_tsv(out_prefix.with_suffix(".fit.tsv"), pd.DataFrame([fit_summary_g]))


if __name__ == "__main__":
    main()
