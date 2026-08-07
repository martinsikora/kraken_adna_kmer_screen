#!/usr/bin/env python3
"""
kraken_screen_lib.py

Shared library for the kraken_adna_kmer_screen workflow.

Combines:
- Utility functions from kraken_abundance.py (prototype 1)
- Taxonomy loaders adapted for the new column schema
  (tax_rank, tax_id, tax_name, tax_ids_descendant)
- DamageAccumulator logic used by
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
import sys
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


def parse_kmer_string_runs(
    s: str,
    exclude_set: set[int],
) -> tuple[list, list, Dict[int, float], int, int]:
    """
    Run-length variant of parse_kmer_string_with_counts: returns the (taxid,
    count) runs as-is instead of expanding them to one entry per k-mer.

    The expanded array is only ever consumed as (a) the first and last max_pos
    entries, (b) a binned histogram, and (c) two scalar sums — all of which are
    derivable from the runs directly. Skipping the expansion removes the largest
    single allocation in the per-read path.

    Returns (tids, counts, taxid_counts, n_kmers, n_unclassified).
    """
    tids: List[int] = []
    counts: List[int] = []
    taxid_counts: Dict[int, float] = {}
    nk = 0
    unc = 0

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
        nk += count
        if taxid == 0:
            unc += count
        if count > 0 and taxid not in exclude_set:
            taxid_counts[taxid] = taxid_counts.get(taxid, 0.0) + count

    return tids, counts, taxid_counts, nk, unc


def end_flags_from_runs(tids, counts, n5: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Unclassified flags for the first n5 and last n5 k-mer positions, with the
    3' array reversed — matching kmers[:n5] and kmers[nk-n5:][::-1].
    """
    head = np.zeros(n5, dtype=np.int64)
    pos = 0
    for t, c in zip(tids, counts):
        if pos >= n5:
            break
        if t == 0:
            head[pos:min(pos + c, n5)] = 1
        pos += c

    tail = np.zeros(n5, dtype=np.int64)
    pos = 0
    for i in range(len(tids) - 1, -1, -1):
        if pos >= n5:
            break
        c = counts[i]
        if tids[i] == 0:
            tail[pos:min(pos + c, n5)] = 1
        pos += c

    return head, tail


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
# Shared accumulator implementation used by the workflow damage scripts.
# ---------------------------------------------------------------------------

def parse_strata_spec(specs: list[str]) -> list[tuple[int, int]]:
    """Parse strata specification strings like '31-40' into (lo, hi) tuples."""
    result: list[tuple[int, int]] = []
    for spec in specs:
        lo, hi = spec.split("-")
        result.append((int(lo), int(hi)))
    return result


class DamageAccumulator:
    """
    Accumulates per-position unclassified k-mer counts across reads.
    Keys may be int taxids or str species names.

    Storage: _5[key] / _3[key] each shape (max_pos, 2)
      column 0 = total k-mers at that position
      column 1 = unclassified k-mers at that position

    When strata are supplied the same counts are additionally kept split by
    read-length stratum in _s5 / _s3, shape (n_strata, max_pos, 2). The pooled
    arrays remain the sum over strata plus any read outside every stratum, so
    every existing consumer -- profile(), raw(), compute_damage_stats(),
    to_dataframe() -- is unaffected. The split exists because a read reaches
    k-mer position j only if its length is at least k + j, so a pooled profile
    silently changes its read composition along the x axis.
    """

    def __init__(self, max_pos: int = 25, strata: list | None = None,
                 kmer_size: int = 0):
        self.max_pos = max_pos
        self.strata = list(strata) if strata else []
        # k is not stated anywhere in the classify file, but follows from any
        # read: n_kmers = length - k + 1. Recording it lets the damage plot
        # label each k-mer index with the read bases it spans.
        self.kmer_size = int(kmer_size)
        self._5: dict = {}
        self._3: dict = {}
        self._n: dict = {}
        self._s5: dict = {}
        self._s3: dict = {}
        self._sn: dict = {}

    def stratum_index(self, read_len: int) -> int:
        """Index of the stratum containing read_len, or -1."""
        for i, (lo, hi) in enumerate(self.strata):
            if lo <= read_len <= hi:
                return i
        return -1

    def _init_key(self, key):
        self._5[key] = np.zeros((self.max_pos, 2), dtype=np.int64)
        self._3[key] = np.zeros((self.max_pos, 2), dtype=np.int64)
        self._n[key] = 0
        ns = len(self.strata)
        if ns:
            self._s5[key] = np.zeros((ns, self.max_pos, 2), dtype=np.int64)
            self._s3[key] = np.zeros((ns, self.max_pos, 2), dtype=np.int64)
            self._sn[key] = np.zeros(ns, dtype=np.int64)

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

    def add_flags(self, key, n5: int, head: np.ndarray, tail: np.ndarray,
                  stratum_idx: int = -1):
        """
        Run-length equivalent of add_read: takes precomputed unclassified flags
        for the first and last n5 positions (3' already reversed) instead of the
        expanded k-mer array. State and arithmetic are identical to add_read.

        stratum_idx, when >= 0, also books the read into that read-length
        stratum; the pooled arrays are updated either way.
        """
        if n5 == 0:
            return
        if key not in self._5:
            self._init_key(key)
        self._n[key] += 1
        self._5[key][:n5, 0] += 1
        self._5[key][:n5, 1] += head
        self._3[key][:n5, 0] += 1
        self._3[key][:n5, 1] += tail
        if stratum_idx >= 0 and key in self._s5:
            self._s5[key][stratum_idx, :n5, 0] += 1
            self._s5[key][stratum_idx, :n5, 1] += head
            self._s3[key][stratum_idx, :n5, 0] += 1
            self._s3[key][stratum_idx, :n5, 1] += tail
            self._sn[key][stratum_idx] += 1

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

    def raw_stratified(self, key) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Return (_s5[key], _s3[key]) — shape (n_strata, max_pos, 2) — or None
        when this accumulator carries no strata for the key.
        """
        if key not in self._s5:
            return None
        return self._s5[key], self._s3[key]

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
                        "kmer_size":         self.kmer_size,
                    })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "taxid", "species_name", "end", "position",
            "n_reads", "n_kmers", "n_unclassified", "frac_unclassified",
            "kmer_size",
        ])

    def to_dataframe_stratified(self, min_reads: int = 100) -> pd.DataFrame:
        """Same as to_dataframe but split by read-length stratum."""
        cols = ["taxid", "species_name", "stratum", "end", "position",
                "n_reads", "n_kmers", "n_unclassified", "frac_unclassified",
                "kmer_size"]
        if not self.strata:
            return pd.DataFrame(columns=cols)
        rows = []
        for key in self.taxids():
            if self._n[key] < min_reads or key not in self._s5:
                continue
            is_int = isinstance(key, int)
            for si, (lo, hi) in enumerate(self.strata):
                nr = int(self._sn[key][si])
                if nr == 0:
                    continue
                for end, arr in (("5prime", self._s5[key][si]),
                                 ("3prime", self._s3[key][si])):
                    for pos in range(self.max_pos):
                        total = int(arr[pos, 0])
                        if total == 0:
                            continue
                        unc = int(arr[pos, 1])
                        rows.append({
                            "taxid":             key if is_int else pd.NA,
                            "species_name":      ""  if is_int else key,
                            "stratum":           f"{lo}-{hi}",
                            "end":               end,
                            "position":          pos,
                            "n_reads":           nr,
                            "n_kmers":           total,
                            "n_unclassified":    unc,
                            "frac_unclassified": unc / total,
                            "kmer_size":         self.kmer_size,
                        })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=cols)


# ---------------------------------------------------------------------------
# Model-based damage estimation (k-mer window deconvolution)
# ---------------------------------------------------------------------------
#
# damage_score measures the drop from the terminal k-mer to a "plateau" further
# in. That plateau is not always reached: a k-mer spans k bases, so for a read
# of length L the k-mer at index j covers bases j..j+k-1, and no k-mer clears
# both termini unless L >= k + 2*(damage decay length). For fragments not much
# longer than k the profile is a U whose two arms are the two read ends, its
# minimum sitting where the window is centred, at j = (L-k)/2. The plateau is
# then still damage-contaminated and damage_score is a lower bound whose size
# depends on read length -- so pooling it across a sample makes it depend on
# that sample's fragment length distribution.
#
# This estimator instead treats the k-mer window as a known convolution and
# inverts it. Per-base mismatch probability for a read of length L:
#
#     d_i = e + a5*exp(-i/lambda5) + a3*exp(-(L-1-i)/lambda3)
#
# where e is the interior floor (sequencing error plus divergence from the
# reference) and the two exponentials are terminal damage. KrakenUniq matches
# k-mers exactly, so a k-mer is classified only if every base in it matches:
#
#     P(k-mer at 5' index j unclassified) = 1 - prod_{i=j}^{j+k-1} (1 - d_i)
#
# The 3'-indexed counts are the same d vector read from the other end, so both
# ends and every read-length stratum are deterministic functions of one
# five-parameter theta. Fitting theta jointly across strata means:
#   * no plateau is required -- e is estimated, not read off a window;
#   * no read-length weighting is required -- theta describes the molecules,
#     not which lengths happened to be sequenced;
#   * short reads become informative rather than a nuisance, since their
#     U-shape constrains a5, a3 and the decay lengths simultaneously;
#   * output is a per-base rate, comparable to mapDamage and to published
#     deamination rates, rather than an unclassified-k-mer fraction.
#
# Emitted alongside damage_score, never in place of it.
#
# Caveat carried in the output: k-mers within a read overlap, so the binomial
# likelihood understates variance. chi2/df is reported and standard errors are
# scaled by sqrt(chi2/df) (quasi-likelihood), which on synthetic controls runs
# around 2-3.

DAMAGE_MODEL_COLUMNS = [
    "taxid", "species_name", "n_reads", "kmer_size",
    "interior_rate", "interior_rate_se",
    "damage_rate_5prime", "damage_rate_5prime_se", "decay_5prime",
    "damage_rate_3prime", "damage_rate_3prime_se", "decay_3prime",
    "terminal_rate_5prime", "terminal_rate_3prime",
    "damage_model_pvalue_5prime", "damage_model_pvalue_3prime",
    "chi2_df", "n_obs", "n_strata_used", "converged",
]

# log-space bounds: (interior, amp5, decay5, amp3, decay3)
_DM_LO = np.log(np.array([1e-7, 1e-7, 0.2, 1e-7, 0.2]))
_DM_HI = np.log(np.array([0.5,  0.9,  60.0, 0.9,  60.0]))


def _stratum_length_weights(
    n_by_pos: np.ndarray, lo: int, hi: int, kmer_size: int, max_pos: int,
) -> list[tuple[int, float]]:
    """
    Recover the read-length composition inside one stratum from its own counts.

    A read contributes to k-mer index j only when it has more than j k-mers, so
    n_by_pos[j] is the number of reads with n_kmers > j and the difference
    between neighbouring positions is the number with exactly that many. No
    extra storage is needed. Reads with n_kmers >= max_pos are censored into the
    last position and are represented by the midpoint of the range the stratum
    still allows.

    Returns [(read_length, n_reads), ...].
    """
    if kmer_size <= 0 or n_by_pos.size == 0 or n_by_pos[0] <= 0:
        return []
    out: list[tuple[int, float]] = []
    for t in range(1, max_pos):
        m = float(n_by_pos[t - 1] - n_by_pos[t])
        if m > 0:
            out.append((t + kmer_size - 1, m))
    censored = float(n_by_pos[max_pos - 1])
    if censored > 0:
        nk_max = hi - kmer_size + 1
        nk_rep = max_pos if nk_max <= max_pos else (max_pos + nk_max) // 2
        out.append((int(nk_rep) + kmer_size - 1, censored))
    return out


def _build_damage_design(
    arr5: np.ndarray, arr3: np.ndarray, strata: list, kmer_size: int,
    max_pos: int, min_stratum_reads: int,
):
    """
    Flatten the stratified count arrays into vectors for the fit.

    Each observation is one (stratum, end, k-mer index). Each observation draws
    on several read lengths, so a second set of vectors maps observation ->
    (read length, weight, window start, window end); predictions are formed per
    length and summed back with bincount.
    """
    obs_n, obs_u = [], []
    m_obs, m_len, m_w, m_lo = [], [], [], []
    lengths: dict[int, int] = {}
    n_strata_used = 0

    for si, (lo_len, hi_len) in enumerate(strata):
        comp5 = _stratum_length_weights(arr5[si, :, 0], lo_len, hi_len,
                                        kmer_size, max_pos)
        if not comp5 or sum(w for _, w in comp5) < min_stratum_reads:
            continue
        n_strata_used += 1
        for end, arr in (("5prime", arr5), ("3prime", arr3)):
            for j in range(max_pos):
                n_j = float(arr[si, j, 0])
                if n_j <= 0:
                    continue
                oid = len(obs_n)
                obs_n.append(n_j)
                obs_u.append(float(arr[si, j, 1]))
                for L, w in comp5:
                    # window must lie inside the read
                    start = j if end == "5prime" else L - kmer_size - j
                    if start < 0 or start + kmer_size > L:
                        continue
                    if L not in lengths:
                        lengths[L] = len(lengths)
                    m_obs.append(oid)
                    m_len.append(lengths[L])
                    m_w.append(w)
                    m_lo.append(start)

    if not obs_n or not m_obs:
        return None

    Lvals = np.zeros(len(lengths), dtype=np.int64)
    for L, idx in lengths.items():
        Lvals[idx] = L

    d = dict(
        obs_n = np.asarray(obs_n, dtype=float),
        obs_u = np.asarray(obs_u, dtype=float),
        m_obs = np.asarray(m_obs, dtype=np.int64),
        m_len = np.asarray(m_len, dtype=np.int64),
        m_w   = np.asarray(m_w,   dtype=float),
        m_lo  = np.asarray(m_lo,  dtype=np.int64),
        Lvals = Lvals,
        Lmax  = int(Lvals.max()),
        kmer_size = kmer_size,
        n_strata_used = n_strata_used,
    )
    # weights actually reaching each observation; the residual normalises by
    # this rather than obs_n, since a length can be dropped by the window test
    d["m_tot"] = np.bincount(d["m_obs"], weights=d["m_w"],
                             minlength=len(obs_n))
    keep = d["m_tot"] > 0
    if not keep.all():
        remap = -np.ones(len(obs_n), dtype=np.int64)
        remap[keep] = np.arange(int(keep.sum()))
        sel = remap[d["m_obs"]] >= 0
        d["obs_n"] = d["obs_n"][keep]
        d["obs_u"] = d["obs_u"][keep]
        d["m_tot"] = d["m_tot"][keep]
        d["m_obs"] = remap[d["m_obs"][sel]]
        d["m_len"] = d["m_len"][sel]
        d["m_w"]   = d["m_w"][sel]
        d["m_lo"]  = d["m_lo"][sel]
    if d["obs_n"].size == 0:
        return None
    return d


def _damage_residual(log_theta: np.ndarray, d: dict) -> np.ndarray:
    """
    Pearson residual between observed and predicted unclassified fractions.

    Standardised by the binomial SD of the prediction, not just weighted by
    sqrt(n), so that chi2/df is interpretable: ~1 under a correct model with
    independent k-mers, and above 1 by the factor that overlapping k-mers
    within a read inflate the variance.
    """
    e, a5, l5, a3, l3 = np.exp(log_theta)
    Lv = d["Lvals"][:, None]
    i = np.arange(d["Lmax"])[None, :]
    dv = e + a5 * np.exp(-i / l5) + a3 * np.exp(-(Lv - 1 - i) / l3)
    dv = np.where(i < Lv, np.clip(dv, 1e-12, 1.0 - 1e-12), 0.0)
    C = np.concatenate(
        [np.zeros((dv.shape[0], 1)), np.cumsum(np.log1p(-dv), axis=1)], axis=1
    )
    lo = d["m_lo"]
    q = 1.0 - np.exp(C[d["m_len"], lo + d["kmer_size"]] - C[d["m_len"], lo])
    u_pred = np.bincount(d["m_obs"], weights=d["m_w"] * q,
                         minlength=d["obs_n"].size)
    pred = np.clip(u_pred / d["m_tot"], 1e-9, 1.0 - 1e-9)
    sd = np.sqrt(pred * (1.0 - pred) / d["obs_n"])
    return (d["obs_u"] / d["obs_n"] - pred) / sd


def fit_damage_model_one(
    arr5: np.ndarray, arr3: np.ndarray, strata: list, kmer_size: int,
    max_pos: int, min_stratum_reads: int = 100,
) -> dict | None:
    """
    Fit the five-parameter damage model to one taxon's stratified counts.

    Returns None when there is not enough stratified data to attempt a fit.
    Returns a dict with converged=False rather than raising when the fit fails,
    so a bad taxon cannot take down a whole sample.
    """
    from scipy.optimize import least_squares

    d = _build_damage_design(arr5, arr3, strata, kmer_size, max_pos,
                             min_stratum_reads)
    if d is None:
        return None
    n_obs = int(d["obs_n"].size)
    n_par = 5
    if n_obs <= n_par:
        return None

    # start from the data: interior floor from the smallest observed fraction,
    # terminal amplitude from the excess at index 0, both per base
    frac = d["obs_u"] / d["obs_n"]
    f_min = float(np.clip(frac.min(), 1e-9, 0.99))
    f_max = float(np.clip(frac.max(), f_min + 1e-9, 0.999))
    e0 = 1.0 - (1.0 - f_min) ** (1.0 / max(kmer_size, 1))
    a0 = max(1.0 - (1.0 - f_max) ** (1.0 / max(kmer_size, 1)) - e0, 1e-5)

    best = None
    for lam0 in (2.0, 6.0):
        p0 = np.log(np.clip(np.array([e0, a0, lam0, a0, lam0]),
                            np.exp(_DM_LO), np.exp(_DM_HI)))
        try:
            fit = least_squares(_damage_residual, p0, args=(d,),
                                bounds=(_DM_LO, _DM_HI), max_nfev=4000)
        except Exception:
            continue
        if best is None or fit.cost < best.cost:
            best = fit
    if best is None:
        return dict(converged=False, n_obs=n_obs,
                    n_strata_used=d["n_strata_used"])

    theta = np.exp(best.x)
    dof = max(n_obs - n_par, 1)
    chi2 = float(np.sum(best.fun ** 2))
    disp = chi2 / dof

    # quasi-likelihood covariance, then delta method back from log space
    se = np.full(n_par, np.nan)
    try:
        JTJ = best.jac.T @ best.jac
        cov_log = np.linalg.inv(JTJ) * disp
        se_log = np.sqrt(np.clip(np.diag(cov_log), 0, None))
        se = theta * se_log
    except np.linalg.LinAlgError:
        pass

    def _p(amp, amp_se):
        if not np.isfinite(amp_se) or amp_se <= 0:
            return np.nan
        return 0.5 * math.erfc((amp / amp_se) / math.sqrt(2.0))

    return dict(
        interior_rate              = float(theta[0]),
        interior_rate_se           = float(se[0]),
        damage_rate_5prime         = float(theta[1]),
        damage_rate_5prime_se      = float(se[1]),
        decay_5prime               = float(theta[2]),
        damage_rate_3prime         = float(theta[3]),
        damage_rate_3prime_se      = float(se[3]),
        decay_3prime               = float(theta[4]),
        terminal_rate_5prime       = float(theta[0] + theta[1]),
        terminal_rate_3prime       = float(theta[0] + theta[3]),
        damage_model_pvalue_5prime = _p(theta[1], se[1]),
        damage_model_pvalue_3prime = _p(theta[3], se[3]),
        chi2_df                    = float(disp),
        n_obs                      = n_obs,
        n_strata_used              = int(d["n_strata_used"]),
        converged                  = bool(best.success),
    )


def fit_damage_models(
    acc: "DamageAccumulator", min_reads: int = 100,
    min_stratum_reads: int = 100, verbose: bool = True,
) -> pd.DataFrame:
    """
    Fit the damage model for every taxon with enough stratified reads.

    Returns an empty frame (correct columns) when the accumulator carries no
    strata or no k-mer size, so callers can write the file unconditionally.
    """
    if not acc.strata or acc.kmer_size <= 0:
        if verbose:
            reason = ("no read-length strata" if not acc.strata
                      else "k-mer size unknown")
            print(f"[damage_model] skipped: {reason}", file=sys.stderr,
                  flush=True)
        return pd.DataFrame(columns=DAMAGE_MODEL_COLUMNS)

    rows, n_fail = [], 0
    for key in acc.taxids():
        if acc.n_reads(key) < min_reads:
            continue
        strat = acc.raw_stratified(key)
        if strat is None:
            continue
        arr5, arr3 = strat
        res = fit_damage_model_one(arr5, arr3, acc.strata, acc.kmer_size,
                                   acc.max_pos, min_stratum_reads)
        if res is None:
            continue
        if not res.get("converged", False):
            n_fail += 1
        is_int = isinstance(key, int)
        rows.append({
            "taxid":        key if is_int else pd.NA,
            "species_name": ""  if is_int else key,
            "n_reads":      acc.n_reads(key),
            "kmer_size":    acc.kmer_size,
            **res,
        })

    if not rows:
        return pd.DataFrame(columns=DAMAGE_MODEL_COLUMNS)
    df = pd.DataFrame(rows)
    for c in DAMAGE_MODEL_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[DAMAGE_MODEL_COLUMNS]
    if verbose:
        med = df["chi2_df"].median()
        print(f"[damage_model] fitted {len(df)} taxa "
              f"({n_fail} not converged), median chi2/df={med:.2f}",
              file=sys.stderr, flush=True)
    return df.sort_values("damage_rate_5prime", ascending=False).reset_index(drop=True)


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
) -> None:
    """
    Save DamageAccumulator state to a compressed npz file.

    Keys are JSON-encoded as a UTF-8 byte array. Per-key arrays are stored
    under 'd5_i', 'd3_i', 'dn_i' (pooled over read length) and, when the
    accumulator carries strata, 's5_i', 's3_i', 'sn_i' (split by stratum).
    """
    keys = list(damage_acc._5.keys())
    save_dict: dict = {
        "keys":     np.frombuffer(json.dumps(keys).encode("utf-8"), dtype=np.uint8),
        "max_pos":  np.array([damage_acc.max_pos], dtype=np.int64),
        "strata":   np.frombuffer(
            json.dumps(damage_acc.strata).encode("utf-8"), dtype=np.uint8),
        "kmer_size": np.array([getattr(damage_acc, "kmer_size", 0)], dtype=np.int64),
    }
    for i, key in enumerate(keys):
        save_dict[f"d5_{i}"] = damage_acc._5[key]
        save_dict[f"d3_{i}"] = damage_acc._3[key]
        save_dict[f"dn_{i}"] = np.array([damage_acc._n[key]], dtype=np.int64)
        if key in damage_acc._s5:
            save_dict[f"s5_{i}"] = damage_acc._s5[key]
            save_dict[f"s3_{i}"] = damage_acc._s3[key]
            save_dict[f"sn_{i}"] = damage_acc._sn[key]
    np.savez_compressed(path, **save_dict)


def load_damage_arrays(path: str | Path) -> DamageAccumulator:
    """Reconstruct a DamageAccumulator from an npz file."""
    data = np.load(path, allow_pickle=False)
    keys    = json.loads(bytes(data["keys"]).decode("utf-8"))
    max_pos = int(data["max_pos"][0])
    strata  = [tuple(x) for x in json.loads(bytes(data["strata"]).decode("utf-8"))]

    kmer_size = int(data["kmer_size"][0]) if "kmer_size" in data.files else 0
    damage_acc = DamageAccumulator(max_pos=max_pos, strata=strata, kmer_size=kmer_size)
    for i, key in enumerate(keys):
        damage_acc._5[key] = data[f"d5_{i}"].copy()
        damage_acc._3[key] = data[f"d3_{i}"].copy()
        damage_acc._n[key] = int(data[f"dn_{i}"][0])
        if f"s5_{i}" in data.files:
            damage_acc._s5[key] = data[f"s5_{i}"].copy()
            damage_acc._s3[key] = data[f"s3_{i}"].copy()
            damage_acc._sn[key] = data[f"sn_{i}"].copy()
    return damage_acc


def merge_damage_accumulators(
    damage_accs: list[DamageAccumulator],
) -> DamageAccumulator:
    """Sum accumulator arrays element-wise across per-unit accumulators."""
    if not damage_accs:
        raise ValueError("damage_accs must be non-empty")

    merged = damage_accs[0]
    for dacc in damage_accs[1:]:
        if getattr(dacc, "kmer_size", 0):
            if not merged.kmer_size:
                merged.kmer_size = dacc.kmer_size
            elif merged.kmer_size != dacc.kmer_size:
                print(f"WARNING: units disagree on k-mer size "
                      f"({merged.kmer_size} vs {dacc.kmer_size}); keeping "
                      f"{merged.kmer_size}", file=sys.stderr)
        for key in dacc.taxids():
            if key not in merged._5:
                merged._init_key(key)
            merged._5[key] += dacc._5[key]
            merged._3[key] += dacc._3[key]
            merged._n[key] += dacc._n[key]
            if key in dacc._s5:
                if key not in merged._s5:
                    merged._s5[key] = np.zeros_like(dacc._s5[key])
                    merged._s3[key] = np.zeros_like(dacc._s3[key])
                    merged._sn[key] = np.zeros(dacc._s5[key].shape[0], dtype=np.int64)
                merged._s5[key] += dacc._s5[key]
                merged._s3[key] += dacc._s3[key]
                merged._sn[key] += dacc._sn[key]
    return merged


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
