#!/usr/bin/env python3
"""
kraken_screen_lib.py

Shared library for the kraken_adna_kmer_screen workflow.

Combines:
- Utility functions from kraken_abundance.py (prototype 1)
- Taxonomy loaders adapted for the new column schema
  (tax_rank, tax_id, tax_name, tax_ids_descendant)
- DamageAccumulator / FractionalAccumulator logic used by
  screen_unit.py and aggregate_sample.py
- Unified kmer string parser for single-pass vectorize+damage accumulation
- Damage array serialization / deserialization helpers
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, Iterator, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import sparse


# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def open_maybe_gzip(path: str | Path, mode: str = "rt"):
    path = str(path)
    if path.endswith(".gz"):
        return gzip.open(path, mode)
    return open(path, mode, encoding="utf-8")


def open_buffered_gzip(path: str | Path, buffer_mb: int = 1) -> io.TextIOWrapper:
    """Open a (possibly gzipped) file with a larger read buffer for efficiency."""
    path = str(path)
    if path.endswith(".gz"):
        raw = io.BufferedReader(
            gzip.open(path, "rb"),
            buffer_size=buffer_mb * 1024 * 1024,
        )
        return io.TextIOWrapper(raw, encoding="utf-8")
    return open(path, "rt", encoding="utf-8", buffering=buffer_mb * 1024 * 1024)


def write_tsv(path: str | Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# KrakenUniq row streaming
# ---------------------------------------------------------------------------

def iter_kraken_rows(path: str | Path) -> Iterator[Tuple[str, str, str, str, str]]:
    with open_maybe_gzip(path, "rt") as handle:
        for row_number, line in enumerate(handle, start=1):
            if line.startswith("#"):
                continue
            row = line.rstrip("\n").split("\t", 4)
            if len(row) != 5:
                raise ValueError(
                    f"{path}: expected 5 columns at row {row_number}, found {len(row)}"
                )
            yield tuple(row)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Kmer string parsers
# ---------------------------------------------------------------------------

def parse_sparse_taxid_counts(
    payload: str,
    exclude_taxids: set[str] | None = None,
) -> Dict[int, float]:
    """Parse 'taxid:count ...' payload into {taxid: total_count}."""
    exclude_taxids = exclude_taxids or set()
    counts: DefaultDict[int, float] = defaultdict(float)
    if not payload:
        return {}
    for token in payload.split():
        if ":" not in token:
            continue
        taxid_text, count_text = token.split(":", 1)
        if taxid_text in exclude_taxids:
            continue
        try:
            taxid = int(taxid_text)
            count = float(count_text)
        except ValueError:
            continue
        if count > 0:
            counts[taxid] += count
    return dict(counts)


def parse_kmer_string(s: str) -> np.ndarray:
    """
    Expand compact kmer string 'taxid:count ...' into a per-kmer array of
    taxids (0 = unclassified). Ambiguous 'A' tokens treated as 0.
    Returns numpy int64 array in 5'->3' order.
    """
    tids: List[int] = []
    counts: List[int] = []
    for token in s.strip().split():
        if ":" not in token:
            continue
        tid, cnt = token.rsplit(":", 1)
        try:
            tids.append(0 if tid == "A" else int(tid))
            counts.append(int(cnt))
        except ValueError:
            pass
    if not tids:
        return np.empty(0, dtype=np.int64)
    return np.repeat(np.array(tids, dtype=np.int64), counts)


def parse_kmer_string_with_counts(
    s: str,
    exclude_set: set[int],
) -> tuple[np.ndarray, Dict[int, float]]:
    """
    Single-pass parser that returns both:
    - per_kmer_array: numpy int64 array for damage accumulation (5'->3')
    - taxid_counts: {taxid: total_count} for NNLS feature vectorization
                    (excludes taxids in exclude_set)

    Combining both outputs from one token pass eliminates redundant string
    parsing when doing the unified single-pass analysis.
    """
    tids: List[int] = []
    counts: List[int] = []
    taxid_counts: Dict[int, float] = {}

    for token in s.strip().split():
        if ":" not in token:
            continue
        tid_str, cnt_str = token.rsplit(":", 1)
        try:
            taxid = 0 if tid_str == "A" else int(tid_str)
            count = int(cnt_str)
        except ValueError:
            continue
        tids.append(taxid)
        counts.append(count)
        if count > 0 and taxid not in exclude_set:
            taxid_counts[taxid] = taxid_counts.get(taxid, 0.0) + count

    if not tids:
        return np.empty(0, dtype=np.int64), taxid_counts
    return np.repeat(np.array(tids, dtype=np.int64), counts), taxid_counts


# ---------------------------------------------------------------------------
# Taxonomy loaders — new column schema
# (tax_rank, tax_id, tax_name, tax_ids_descendant)
# ---------------------------------------------------------------------------

def load_species_membership_v2(
    path: str | Path,
) -> Tuple[Dict[int, Tuple[int, str]], Dict[int, str]]:
    """
    Load species.tax_ids.tsv.gz with new column schema.

    Returns:
        child_to_species : child_taxid -> (species_taxid, species_name)
        species_id_to_name: species_taxid -> species_name
    """
    df = pd.read_csv(
        path,
        sep="\t",
        compression="infer",
        dtype={"tax_id": "Int64", "tax_ids_descendant": "Int64"},
    )
    df = df[df["tax_rank"] == "species"].copy()

    child_to_species: Dict[int, Tuple[int, str]] = {}
    species_id_to_name: Dict[int, str] = {}

    for row in df.itertuples(index=False):
        try:
            child_id = int(row.tax_ids_descendant)
            species_taxid = int(row.tax_id)
        except (TypeError, ValueError):
            continue
        species_name = str(row.tax_name)
        child_to_species[child_id] = (species_taxid, species_name)
        species_id_to_name[species_taxid] = species_name

    return child_to_species, species_id_to_name


def load_genus_membership_v2(
    path: str | Path,
) -> Dict[int, Tuple[int, str]]:
    """
    Load genus.tax_ids.tsv.gz with new column schema.

    Returns:
        child_to_genus: child_taxid -> (genus_taxid, genus_name)
    """
    child_to_genus: Dict[int, Tuple[int, str]] = {}
    with open_maybe_gzip(path, "rt") as fh:
        fh.readline()  # skip header
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            tax_rank, tax_id_text, tax_name, child_id_text = parts[:4]
            if tax_rank != "genus":
                continue
            try:
                genus_taxid = int(tax_id_text)
                child_taxid = int(child_id_text)
            except ValueError:
                continue
            if child_taxid not in child_to_genus:
                child_to_genus[child_taxid] = (genus_taxid, tax_name)
    return child_to_genus


# ---------------------------------------------------------------------------
# Legacy taxonomy loaders (kept for backward compatibility)
# ---------------------------------------------------------------------------

def load_seqid_to_taxid(path: str | Path) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    with open(path, "rt", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row in reader:
            if len(row) != 2:
                continue
            seqid, taxid = row
            mapping[seqid] = int(taxid)
    return mapping


# ---------------------------------------------------------------------------
# Sparse matrix / vector utilities
# ---------------------------------------------------------------------------

def build_sparse_matrix(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    shape: Tuple[int, int],
) -> sparse.csr_matrix:
    matrix = sparse.coo_matrix(
        (
            data.astype(np.float64, copy=False),
            (rows.astype(np.int32, copy=False), cols.astype(np.int32, copy=False)),
        ),
        shape=shape,
    )
    return matrix.tocsr()


def csc_sidecar_path(path: str | Path) -> Path:
    path = Path(path)
    if path.suffix == ".npz":
        return path.with_name(path.stem + ".csc.npz")
    return path.with_name(path.name + ".csc.npz")


def save_sparse_matrix(
    path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    shape: Tuple[int, int],
) -> None:
    csr_matrix = build_sparse_matrix(rows, cols, data, shape)
    sparse.save_npz(path, csr_matrix)
    sparse.save_npz(csc_sidecar_path(path), csr_matrix.tocsc())


def load_sparse_matrix(path: str | Path) -> sparse.csr_matrix:
    return sparse.load_npz(path).tocsr()


def load_sparse_matrix_for_column_slicing(path: str | Path) -> sparse.csc_matrix:
    sidecar = csc_sidecar_path(path)
    if sidecar.exists():
        return sparse.load_npz(sidecar).tocsc()
    csr_matrix = sparse.load_npz(path).tocsr()
    csc_matrix = csr_matrix.tocsc()
    sparse.save_npz(sidecar, csc_matrix)
    return csc_matrix


def save_sparse_vector(
    path: str | Path,
    indices: np.ndarray,
    data: np.ndarray,
    size: int,
) -> None:
    vector = sparse.csr_matrix(
        (
            data.astype(np.float64, copy=False),
            (
                np.zeros(len(indices), dtype=np.int32),
                indices.astype(np.int32, copy=False),
            ),
        ),
        shape=(1, size),
    )
    sparse.save_npz(path, vector)


def load_sparse_vector(path: str | Path) -> sparse.csr_matrix:
    return sparse.load_npz(path).tocsr()


def read_feature_index(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def build_feature_lookup(path: str | Path) -> Dict[int, int]:
    df = read_feature_index(path)
    return {int(row.taxid): int(row.feature_index) for row in df.itertuples(index=False)}


# ---------------------------------------------------------------------------
# aDNA damage accumulation classes
# Shared accumulator implementation used by workflow damage scripts.
# ---------------------------------------------------------------------------

def parse_strata_spec(specs: list[str]) -> list[tuple[int, int]]:
    """Parse strata specification strings like '31-40' into (lo, hi) tuples."""
    result: list[tuple[int, int]] = []
    for spec in specs:
        lo, hi = spec.split("-")
        result.append((int(lo), int(hi)))
    return result


class FractionalAccumulator:
    """
    Accumulates fractional-position k-mer counts per read-length stratum.

    Key may be an int taxid or a str species name.

    Per key:
      frac_total  shape (n_strata, n_bins)  total k-mers per fractional bin
      frac_unc    shape (n_strata, n_bins)  unclassified k-mers per bin
      read_counts shape (n_strata,)         reads per stratum
    """

    def __init__(self, strata: list[tuple[int, int]], n_bins: int = 100):
        self.strata = strata
        self.n_bins = n_bins
        self._total: dict = {}
        self._unc:   dict = {}
        self._reads: dict = {}

    def _init(self, key):
        ns = len(self.strata)
        self._total[key] = np.zeros((ns, self.n_bins), dtype=np.int64)
        self._unc[key]   = np.zeros((ns, self.n_bins), dtype=np.int64)
        self._reads[key] = np.zeros(ns,                dtype=np.int64)

    def _stratum_idx(self, length: int) -> int:
        for i, (lo, hi) in enumerate(self.strata):
            if lo <= length <= hi:
                return i
        return -1

    def add_read(self, key, read_len: int, kmers: np.ndarray):
        s = self._stratum_idx(read_len)
        if s < 0:
            return
        nk = len(kmers)
        if nk < 2:
            return
        if key not in self._total:
            self._init(key)
        self._reads[key][s] += 1
        fracs = np.arange(nk) / (nk - 1)
        bis   = np.minimum((fracs * self.n_bins).astype(int), self.n_bins - 1)
        np.add.at(self._total[key][s], bis, 1)
        np.add.at(self._unc[key][s],   bis, (kmers == 0).astype(np.int64))

    def to_dataframe(self, min_reads: int = 1) -> pd.DataFrame:
        """
        Export fractional profiles for keys with at least `min_reads` total reads
        across all configured strata. Per-stratum rows are emitted whenever a
        stratum has at least one read.
        """
        rows = []
        for key in self._total:
            is_int = isinstance(key, int)
            total_reads = int(self._reads[key].sum())
            if total_reads < min_reads:
                continue
            for s_idx, (lo, hi) in enumerate(self.strata):
                n_reads = int(self._reads[key][s_idx])
                if n_reads == 0:
                    continue
                tot_arr = self._total[key][s_idx]
                unc_arr = self._unc[key][s_idx]
                for b in range(self.n_bins):
                    tot = int(tot_arr[b])
                    if tot == 0:
                        continue
                    unc = int(unc_arr[b])
                    rows.append({
                        "taxid":             key if is_int else pd.NA,
                        "species_name":      ""  if is_int else key,
                        "stratum":           f"{lo}-{hi}",
                        "bin":               b,
                        "n_reads":           n_reads,
                        "n_total":           tot,
                        "n_unclassified":    unc,
                        "frac_unclassified": unc / tot,
                    })
        if not rows:
            return pd.DataFrame(columns=[
                "taxid", "species_name", "stratum", "bin",
                "n_reads", "n_total", "n_unclassified", "frac_unclassified",
            ])
        return pd.DataFrame(rows)


class DamageAccumulator:
    """
    Accumulates per-position unclassified k-mer counts across reads.
    Keys may be int taxids or str species names.

    Storage: _5[key] / _3[key] each shape (max_pos, 2)
      column 0 = total k-mers at that position
      column 1 = unclassified k-mers at that position
    """

    def __init__(self, max_pos: int = 25):
        self.max_pos = max_pos
        self._5: dict = {}
        self._3: dict = {}
        self._n: dict = {}

    def _init_key(self, key):
        self._5[key] = np.zeros((self.max_pos, 2), dtype=np.int64)
        self._3[key] = np.zeros((self.max_pos, 2), dtype=np.int64)
        self._n[key] = 0

    def add_read(self, key, kmers: np.ndarray):
        nk = len(kmers)
        if nk == 0:
            return
        if key not in self._5:
            self._init_key(key)
        self._n[key] += 1
        n5 = min(nk, self.max_pos)

        # 5' end: first n5 k-mers
        self._5[key][:n5, 0] += 1
        self._5[key][:n5, 1] += (kmers[:n5] == 0).astype(np.int64)

        # 3' end: last n5 k-mers in reverse
        tail = kmers[nk - n5:][::-1]
        self._3[key][:n5, 0] += 1
        self._3[key][:n5, 1] += (tail == 0).astype(np.int64)

    def taxids(self):
        return list(self._5.keys())

    def n_reads(self, key) -> int:
        return self._n.get(key, 0)

    def profile(self, key) -> tuple[np.ndarray, np.ndarray]:
        """Return (frac_5prime, frac_3prime) arrays of shape (max_pos,)."""
        def safe_frac(arr):
            t = arr[:, 0].astype(float)
            u = arr[:, 1].astype(float)
            return np.where(t > 0, u / t, np.nan)
        return safe_frac(self._5[key]), safe_frac(self._3[key])

    def raw(self, key) -> tuple[np.ndarray, np.ndarray]:
        """Return (_5[key], _3[key]) — shape (max_pos, 2) count arrays."""
        return self._5[key], self._3[key]

    def to_dataframe(self, min_reads: int = 100) -> pd.DataFrame:
        rows = []
        for key in self.taxids():
            if self._n[key] < min_reads:
                continue
            is_int = isinstance(key, int)
            f5, f3 = self.profile(key)
            for pos in range(self.max_pos):
                for end, frac, arr in [
                    ("5prime", f5, self._5[key]),
                    ("3prime", f3, self._3[key]),
                ]:
                    total = int(arr[pos, 0])
                    if total == 0:
                        continue
                    rows.append({
                        "taxid":             key if is_int else pd.NA,
                        "species_name":      ""  if is_int else key,
                        "end":               end,
                        "position":          pos,
                        "n_reads":           self._n[key],
                        "n_kmers":           total,
                        "n_unclassified":    int(arr[pos, 1]),
                        "frac_unclassified": frac[pos],
                    })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "taxid", "species_name", "end", "position",
            "n_reads", "n_kmers", "n_unclassified", "frac_unclassified",
        ])


# ---------------------------------------------------------------------------
# Damage scoring
# ---------------------------------------------------------------------------

def find_adaptive_plateau(
    frac:         np.ndarray,
    raw_total:    np.ndarray,
    raw_unc:      np.ndarray,
    search_start: int   = 3,
    search_end:   int   = 10,
    min_window:   int   = 3,
    noise_factor: float = 2.0,
) -> tuple[int, int, float, int, int]:
    """
    Find the flattest window of positions within [search_start, search_end]
    as the undamaged plateau baseline.

    Returns (plateau_start, plateau_end, plateau_mean, n_valid_positions, n_windows_tested).
    Falls back to the lowest-mean window if no window passes the noise filter.
    """
    valid_positions = [
        p for p in range(search_start, min(search_end + 1, len(frac)))
        if raw_total[p] > 0 and np.isfinite(frac[p])
    ]
    if len(valid_positions) < min_window:
        # Expand search to all available positions
        valid_positions = [
            p for p in range(len(frac))
            if raw_total[p] > 0 and np.isfinite(frac[p])
        ]
    if len(valid_positions) < 2:
        ps = valid_positions[0] if valid_positions else search_start
        pe = ps
        plateau_val = float(frac[ps]) if valid_positions else np.nan
        return ps, pe, plateau_val, len(valid_positions), 0

    # If we have too few positions for min_window, use all available positions
    # as a single fallback plateau window instead of indexing past bounds.
    if len(valid_positions) < min_window:
        ps = valid_positions[0]
        pe = valid_positions[-1]
        n_tot = sum(int(raw_total[p]) for p in valid_positions)
        n_unc = sum(int(raw_unc[p]) for p in valid_positions)
        plateau_val = (n_unc / n_tot) if n_tot > 0 else np.nan
        return ps, pe, plateau_val, len(valid_positions), 1

    best_ps, best_pe = valid_positions[0], valid_positions[min_window - 1]
    best_mean = np.nan
    best_std  = np.inf
    n_windows = 0
    fallback_ps, fallback_pe, fallback_mean = best_ps, best_pe, np.nan

    for start_idx in range(len(valid_positions) - min_window + 1):
        for end_idx in range(start_idx + min_window - 1, len(valid_positions)):
            window_pos = valid_positions[start_idx : end_idx + 1]
            n_windows += 1

            n_tot = sum(int(raw_total[p]) for p in window_pos)
            n_unc = sum(int(raw_unc[p])   for p in window_pos)
            pooled_mean = n_unc / n_tot if n_tot > 0 else np.nan
            if not np.isfinite(pooled_mean):
                continue

            obs_std = float(np.std([frac[p] for p in window_pos]))
            exp_std = float(np.sqrt(
                np.mean([
                    pooled_mean * (1 - pooled_mean) / max(int(raw_total[p]), 1)
                    for p in window_pos
                ])
            ))

            # Track the globally lowest-mean window as fallback
            if np.isnan(fallback_mean) or pooled_mean < fallback_mean:
                fallback_ps   = window_pos[0]
                fallback_pe   = window_pos[-1]
                fallback_mean = pooled_mean

            # Reject noisy (zig-zagging) windows
            if exp_std > 0 and obs_std > noise_factor * exp_std:
                continue

            # Select window with lowest pooled mean; ties broken by lowest obs_std
            if np.isnan(best_mean) or pooled_mean < best_mean or (
                pooled_mean == best_mean and obs_std < best_std
            ):
                best_ps   = window_pos[0]
                best_pe   = window_pos[-1]
                best_mean = pooled_mean
                best_std  = obs_std

    if np.isnan(best_mean):
        # All windows rejected by noise filter — use fallback
        return fallback_ps, fallback_pe, fallback_mean, len(valid_positions), n_windows

    return best_ps, best_pe, best_mean, len(valid_positions), n_windows


def _score_with_uncertainty(
    arr: np.ndarray,
    plateau_start: int,
    plateau_end: int,
    ci_z: float = 1.96,
) -> dict:
    """
    Compute damage score = pos0_rate - plateau_rate with binomial uncertainty.
    """
    from math import erfc, sqrt

    if arr[0, 0] == 0:
        return {"score": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "pvalue": np.nan}

    n0 = int(arr[0, 0])
    k0 = int(arr[0, 1])
    p0 = k0 / n0

    plateau_positions = [
        p for p in range(plateau_start, plateau_end + 1)
        if p < len(arr) and arr[p, 0] > 0
    ]
    if not plateau_positions:
        return {"score": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "pvalue": np.nan}

    n_plat = sum(int(arr[p, 0]) for p in plateau_positions)
    k_plat = sum(int(arr[p, 1]) for p in plateau_positions)
    p_plat = k_plat / n_plat if n_plat > 0 else np.nan

    if not np.isfinite(p_plat):
        return {"score": np.nan, "se": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "pvalue": np.nan}

    score = p0 - p_plat
    var0   = p0 * (1 - p0) / n0
    var_pl = p_plat * (1 - p_plat) / n_plat if n_plat > 0 else 0.0
    se = sqrt(var0 + var_pl)
    ci_lo = score - ci_z * se
    ci_hi = score + ci_z * se

    n_total_pool = n0 + n_plat
    k_total_pool = k0 + k_plat
    p_pool = k_total_pool / n_total_pool if n_total_pool > 0 else 0.0
    denom = sqrt(p_pool * (1 - p_pool) * (1.0 / n0 + 1.0 / n_plat)) if n_plat > 0 else 0.0
    z = (p0 - p_plat) / denom if denom > 0 else 0.0
    pvalue = 0.5 * erfc(z / sqrt(2))

    return {"score": score, "se": se, "ci_lo": ci_lo, "ci_hi": ci_hi, "pvalue": pvalue}


def compute_damage_stats(
    acc: DamageAccumulator,
    min_reads: int = 100,
    adaptive_plateau: bool = True,
    plateau_start: int = 2,
    plateau_end: int = 5,
    plateau_search_start: int = 3,
    plateau_search_end: int = 10,
    min_plateau_window: int = 3,
    plateau_noise_factor: float = 2.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        stats_df  — per-key/end: pos0, plateau window, score + uncertainty
        global_df — per-key: primary 5' damage score + uncertainty, 3' score
    """
    stat_rows, global_rows = [], []

    for key in acc.taxids():
        if acc.n_reads(key) < min_reads:
            continue
        is_int   = isinstance(key, int)
        f5, f3   = acc.profile(key)
        arr5, arr3 = acc.raw(key)
        n_reads  = acc.n_reads(key)
        end_data: dict = {}

        for end, frac, arr in [("5prime", f5, arr5), ("3prime", f3, arr3)]:
            if adaptive_plateau:
                ps, pe, plateau_val, _, _ = find_adaptive_plateau(
                    frac, arr[:, 0], arr[:, 1],
                    search_start = plateau_search_start,
                    search_end   = plateau_search_end,
                    min_window   = min_plateau_window,
                    noise_factor = plateau_noise_factor,
                )
            else:
                ps, pe = plateau_start, plateau_end
                positions = [p for p in range(ps, pe + 1) if p < len(arr) and arr[p, 0] > 0]
                n_plat = sum(arr[p, 0] for p in positions)
                k_plat = sum(arr[p, 1] for p in positions)
                plateau_val = k_plat / n_plat if n_plat > 0 else np.nan

            stats = _score_with_uncertainty(arr, ps, pe)
            end_data[end] = stats

            pos0 = float(arr[0, 1] / arr[0, 0]) if arr[0, 0] > 0 else np.nan
            stat_rows.append({
                "taxid":                key if is_int else pd.NA,
                "species_name":         ""  if is_int else key,
                "end":                  end,
                "pos0_frac_unc":        pos0,
                "plateau_frac_unc":     plateau_val,
                "plateau_pos_start":    ps,
                "plateau_pos_end":      pe,
                "damage_score":         stats["score"],
                "damage_score_se":      stats["se"],
                "damage_score_ci95_lo": stats["ci_lo"],
                "damage_score_ci95_hi": stats["ci_hi"],
                "damage_pvalue":        stats["pvalue"],
                "n_reads":              n_reads,
            })

        s5 = end_data.get("5prime", {})
        s3 = end_data.get("3prime", {})
        global_rows.append({
            "taxid":                key if is_int else pd.NA,
            "species_name":         ""  if is_int else key,
            "n_reads":              n_reads,
            "damage_score":         s5.get("score",  np.nan),
            "damage_score_se":      s5.get("se",     np.nan),
            "damage_score_ci95_lo": s5.get("ci_lo",  np.nan),
            "damage_score_ci95_hi": s5.get("ci_hi",  np.nan),
            "damage_pvalue":        s5.get("pvalue", np.nan),
            "damage_score_3prime":  s3.get("score",  np.nan),
        })

    stats_df  = pd.DataFrame(stat_rows) if stat_rows else pd.DataFrame(columns=[
        "taxid", "species_name", "end", "pos0_frac_unc", "plateau_frac_unc",
        "plateau_pos_start", "plateau_pos_end", "damage_score", "damage_score_se",
        "damage_score_ci95_lo", "damage_score_ci95_hi", "damage_pvalue", "n_reads",
    ])
    global_df = pd.DataFrame(global_rows) if global_rows else pd.DataFrame(columns=[
        "taxid", "species_name", "n_reads", "damage_score", "damage_score_se",
        "damage_score_ci95_lo", "damage_score_ci95_hi", "damage_pvalue", "damage_score_3prime",
    ])
    if not stats_df.empty:
        stats_df = stats_df.sort_values("damage_score", ascending=False).reset_index(drop=True)
    if not global_df.empty:
        global_df = global_df.sort_values("damage_score", ascending=False).reset_index(drop=True)
    return stats_df, global_df


# ---------------------------------------------------------------------------
# Damage accumulator serialization / deserialization
# ---------------------------------------------------------------------------

def save_damage_arrays(
    path: str | Path,
    damage_acc: DamageAccumulator,
    frac_acc: FractionalAccumulator,
) -> None:
    """
    Save DamageAccumulator + FractionalAccumulator state to a compressed npz file.

    Keys are JSON-encoded as a UTF-8 byte array. Per-key arrays are stored
    under 'd5_i', 'd3_i', 'dn_i' (damage) and 'ft_i', 'fu_i', 'fr_i' (fractional).
    """
    keys = list(damage_acc._5.keys())
    save_dict: dict = {
        "keys":     np.frombuffer(json.dumps(keys).encode("utf-8"), dtype=np.uint8),
        "max_pos":  np.array([damage_acc.max_pos], dtype=np.int64),
        "n_bins":   np.array([frac_acc.n_bins],    dtype=np.int64),
        "strata":   np.frombuffer(json.dumps(frac_acc.strata).encode("utf-8"), dtype=np.uint8),
    }
    for i, key in enumerate(keys):
        save_dict[f"d5_{i}"] = damage_acc._5[key]
        save_dict[f"d3_{i}"] = damage_acc._3[key]
        save_dict[f"dn_{i}"] = np.array([damage_acc._n[key]], dtype=np.int64)
        if key in frac_acc._total:
            save_dict[f"ft_{i}"] = frac_acc._total[key]
            save_dict[f"fu_{i}"] = frac_acc._unc[key]
            save_dict[f"fr_{i}"] = frac_acc._reads[key]
        else:
            ns = len(frac_acc.strata)
            save_dict[f"ft_{i}"] = np.zeros((ns, frac_acc.n_bins), dtype=np.int64)
            save_dict[f"fu_{i}"] = np.zeros((ns, frac_acc.n_bins), dtype=np.int64)
            save_dict[f"fr_{i}"] = np.zeros(ns, dtype=np.int64)
    np.savez_compressed(path, **save_dict)


def load_damage_arrays(
    path: str | Path,
) -> tuple[DamageAccumulator, FractionalAccumulator]:
    """
    Reconstruct DamageAccumulator + FractionalAccumulator from npz file.
    """
    data = np.load(path, allow_pickle=False)
    keys     = json.loads(bytes(data["keys"]).decode("utf-8"))
    max_pos  = int(data["max_pos"][0])
    n_bins   = int(data["n_bins"][0])
    strata   = [tuple(s) for s in json.loads(bytes(data["strata"]).decode("utf-8"))]

    damage_acc = DamageAccumulator(max_pos=max_pos)
    frac_acc   = FractionalAccumulator(strata=strata, n_bins=n_bins)

    for i, key in enumerate(keys):
        damage_acc._5[key] = data[f"d5_{i}"].copy()
        damage_acc._3[key] = data[f"d3_{i}"].copy()
        damage_acc._n[key] = int(data[f"dn_{i}"][0])
        frac_acc._total[key] = data[f"ft_{i}"].copy()
        frac_acc._unc[key]   = data[f"fu_{i}"].copy()
        frac_acc._reads[key] = data[f"fr_{i}"].copy()

    return damage_acc, frac_acc


def merge_damage_accumulators(
    damage_accs: list[DamageAccumulator],
    frac_accs:   list[FractionalAccumulator],
) -> tuple[DamageAccumulator, FractionalAccumulator]:
    """
    Sum accumulator arrays element-wise across multiple per-unit accumulators.
    Returns a single merged DamageAccumulator + FractionalAccumulator.
    """
    if not damage_accs:
        raise ValueError("damage_accs must be non-empty")

    merged_dmg  = damage_accs[0]
    merged_frac = frac_accs[0]

    for dacc, facc in zip(damage_accs[1:], frac_accs[1:]):
        for key in dacc.taxids():
            if key not in merged_dmg._5:
                merged_dmg._init_key(key)
            merged_dmg._5[key] += dacc._5[key]
            merged_dmg._3[key] += dacc._3[key]
            merged_dmg._n[key] += dacc._n[key]
        for key in facc._total:
            if key not in merged_frac._total:
                merged_frac._init(key)
            merged_frac._total[key] += facc._total[key]
            merged_frac._unc[key]   += facc._unc[key]
            merged_frac._reads[key] += facc._reads[key]

    return merged_dmg, merged_frac


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def normalize_counts(counts: Dict[int, float]) -> Dict[int, float]:
    total = float(sum(counts.values()))
    if total <= 0:
        return {}
    return {taxid: value / total for taxid, value in counts.items() if value > 0}


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("Value must be a positive finite number")
    return parsed
