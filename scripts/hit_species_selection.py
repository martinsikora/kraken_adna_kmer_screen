#!/usr/bin/env python3
"""
Helpers for selecting per-sample hit species from all_samples.summary.tsv(.gz).
"""

from __future__ import annotations

import pandas as pd


_USECOLS = [
    "sample_id",
    "species_name",
    "hit_criteria_flag",
    "damage_pvalue",
]

TOKEN_ALIASES = {
    "within_genus_abundance": "within_genus_relative_abundance",
}


def _normalize_required_tokens(tokens: list[str] | None) -> list[str]:
    if not tokens:
        return []
    out: list[str] = []
    for raw in tokens:
        tok = str(raw).strip()
        if tok == "":
            continue
        tok = TOKEN_ALIASES.get(tok, tok)
        if tok not in out:
            out.append(tok)
    return out


def _has_all_flag_tokens(values: pd.Series, required_tokens: list[str]) -> pd.Series:
    required = _normalize_required_tokens(required_tokens)
    if not required:
        return pd.Series(True, index=values.index, dtype=bool)
    required_set = set(required)
    return values.fillna("").astype(str).str.split(";").apply(
        lambda xs: required_set.issubset(
            {x.strip() for x in xs if isinstance(x, str) and x.strip() != ""}
        )
    )


def _read_sample_rows(
    hits_path: str,
    sample_id: str,
    chunksize: int = 250_000,
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        hits_path,
        sep="\t",
        dtype={"sample_id": str},
        usecols=lambda c: c in _USECOLS,
        chunksize=chunksize,
    ):
        sub = chunk[chunk["sample_id"] == sample_id]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def select_hit_species(
    hits_path: str,
    sample_id: str,
    required_flag_tokens: list[str] | None = None,
    max_keys: int | None = 200,
) -> tuple[list[str], dict[str, int]]:
    """
    Return selected species names for one sample from a hit summary table.

    Selection order:
      1) Filter to sample_id rows.
      2) Keep rows where hit_criteria_flag contains all required tokens.
         If required_flag_tokens is empty, taxa are selected without
         token filtering.
      3) De-duplicate by species_name.
      4) Rank by lowest damage_pvalue where available, then species_name.
      5) Apply max_keys cap when > 0.
    """
    sample_id = str(sample_id)
    hdf = _read_sample_rows(hits_path, sample_id=sample_id)
    if hdf.empty:
        return [], {
            "n_sample_rows": 0,
            "n_selected_rows": 0,
            "n_selected_species": 0,
            "n_truncated": 0,
        }

    if "species_name" not in hdf.columns:
        return [], {
            "n_sample_rows": len(hdf),
            "n_selected_rows": 0,
            "n_selected_species": 0,
            "n_truncated": 0,
        }

    hdf["species_name"] = hdf["species_name"].fillna("").astype(str).str.strip()
    hdf = hdf[hdf["species_name"] != ""].copy()
    if hdf.empty:
        return [], {
            "n_sample_rows": 0,
            "n_selected_rows": 0,
            "n_selected_species": 0,
            "n_truncated": 0,
        }

    required = _normalize_required_tokens(required_flag_tokens)
    if "hit_criteria_flag" not in hdf.columns:
        return [], {
            "n_sample_rows": len(hdf),
            "n_selected_rows": 0,
            "n_selected_species": 0,
            "n_truncated": 0,
        }
    mask = _has_all_flag_tokens(hdf["hit_criteria_flag"], required_tokens=required)

    sel = hdf[mask].copy()
    if sel.empty:
        return [], {
            "n_sample_rows": len(hdf),
            "n_selected_rows": 0,
            "n_selected_species": 0,
            "n_truncated": 0,
        }

    if "damage_pvalue" in sel.columns:
        sel["damage_pvalue_num"] = pd.to_numeric(sel["damage_pvalue"], errors="coerce")
        ranked = (
            sel.groupby("species_name", as_index=False)["damage_pvalue_num"]
            .min()
            .sort_values(["damage_pvalue_num", "species_name"], ascending=[True, True], na_position="last")
        )
    else:
        ranked = (
            sel[["species_name"]]
            .drop_duplicates()
            .sort_values(["species_name"])
        )

    species = ranked["species_name"].astype(str).tolist()
    n_truncated = 0
    if max_keys is not None and int(max_keys) > 0 and len(species) > int(max_keys):
        n_truncated = len(species) - int(max_keys)
        species = species[: int(max_keys)]

    return species, {
        "n_sample_rows": len(hdf),
        "n_selected_rows": len(sel),
        "n_selected_species": len(ranked),
        "n_truncated": n_truncated,
    }
