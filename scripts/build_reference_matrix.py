#!/usr/bin/env python3
"""
build_reference_matrix.py

Build a species-level sparse reference matrix from a KrakenUniq database
fingerprint TSV. Run once per database; outputs are reused across all samples.

Outputs (all written to --out-prefix.*):
  .matrix.npz      SciPy sparse CSR matrix (species × features)
  .matrix.csc.npz  CSC sidecar for efficient column slicing during fitting
  .species.tsv     Row metadata (species_index, species_taxid, species_name, ...)
  .features.tsv    Column metadata (feature_index, taxid)
  .summary.tsv     Build diagnostics

This script is self-contained and has no dependency on kraken_screen_lib.
A few helpers (iter_kraken_rows, save_sparse_matrix, load_species_membership_v2,
load_seqid_to_taxid) are duplicated from kraken_screen_lib on purpose so the
builder can be copied next to a database and run stand-alone — do not
"deduplicate" them back into the library.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, List, Tuple

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


# ---------------------------------------------------------------------------
# Taxonomy loaders
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


def load_species_membership_v2(
    path: str | Path,
) -> Tuple[Dict[int, Tuple[int, str]], Dict[int, str]]:
    """
    Load species.tax_ids.tsv.gz (columns: tax_rank, tax_id, tax_name, tax_ids_descendant).

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
# Sparse matrix output
# ---------------------------------------------------------------------------

def save_sparse_matrix(
    path: str | Path,
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    shape: Tuple[int, int],
) -> None:
    path = Path(path)
    csr = sparse.coo_matrix(
        (
            data.astype(np.float64, copy=False),
            (rows.astype(np.int32, copy=False), cols.astype(np.int32, copy=False)),
        ),
        shape=shape,
    ).tocsr()
    sparse.save_npz(path, csr)
    # CSC sidecar: foo.matrix.npz -> foo.matrix.csc.npz
    csc_path = path.with_name(path.stem + ".csc.npz")
    sparse.save_npz(csc_path, csr.tocsc())


# ---------------------------------------------------------------------------
# TSV output helper
# ---------------------------------------------------------------------------

def write_tsv(path: str | Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, sep="\t", index=False)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a species-level sparse reference matrix from a KrakenUniq database fingerprint TSV."
    )
    parser.add_argument("--database-tsv",   required=True)
    parser.add_argument("--seqid-map",      required=True)
    parser.add_argument("--species-taxids", required=True)
    parser.add_argument("--out-prefix",     required=True)
    parser.add_argument(
        "--exclude-taxid",
        action="append",
        type=int,
        default=[],
        help="Exclude a taxid from the reference matrix build. Repeatable.",
    )
    parser.add_argument(
        "--exclude-taxid-file",
        action="append",
        default=[],
        help="Read taxids to exclude from a file (one per line). Repeatable.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10000,
        help="Report progress every N rows (0 to disable).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_excluded_taxids(args: argparse.Namespace) -> tuple[set[int], set[int]]:
    user_excluded: set[int] = set(int(v) for v in args.exclude_taxid)
    for path in args.exclude_taxid_file:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            for token in line.replace(",", " ").split():
                user_excluded.add(int(token))
    applied = set(user_excluded)
    applied.add(0)  # always exclude unclassified
    return user_excluded, applied


def accumulate_weighted_taxid_counts(
    payload: str,
    weight: float,
    destination: Dict[int, float],
    exclude_taxids: set[int],
) -> tuple[set[int], int, float]:
    parsed: List[Tuple[int, float]] = []
    total = 0.0
    for token in payload.split():
        if ":" not in token:
            continue
        taxid_text, count_text = token.split(":", 1)
        try:
            taxid = int(taxid_text)
            count = float(count_text)
        except ValueError:
            continue
        if count <= 0:
            continue
        parsed.append((taxid, count))
        total += count

    if total <= 0:
        return set(), 0, 0.0

    scale = weight / total
    observed: set[int] = set()
    excl_occ = 0
    excl_mass = 0.0
    for taxid, count in parsed:
        if taxid in exclude_taxids:
            excl_occ  += 1
            excl_mass += count * scale
            continue
        destination[taxid] = destination.get(taxid, 0.0) + (count * scale)
        observed.add(taxid)
    return observed, excl_occ, excl_mass


def report_progress(
    *,
    rows_processed: int,
    rows_skipped_missing_mapping: int,
    rows_skipped_empty: int,
    n_species: int,
    n_features: int,
    start_time: float,
    final: bool = False,
) -> None:
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    rps = rows_processed / elapsed if rows_processed > 0 else 0.0
    label = "done" if final else "progress"
    print(
        f"[build_reference_matrix] {label}: rows={rows_processed:,} "
        f"elapsed_s={elapsed:.1f} rows_per_s={rps:.1f} "
        f"species={n_species:,} features={n_features:,} "
        f"skipped_mapping={rows_skipped_missing_mapping:,} "
        f"skipped_empty={rows_skipped_empty:,}",
        file=sys.stderr,
        flush=True,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    out_prefix = Path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()

    user_excluded, excluded_taxids = load_excluded_taxids(args)
    seqid_to_taxid = load_seqid_to_taxid(args.seqid_map)
    child_to_species, species_name_lookup = load_species_membership_v2(args.species_taxids)

    species_feature_weights: DefaultDict[int, Dict[int, float]] = defaultdict(dict)
    species_total_weight: Counter[int] = Counter()
    species_sequence_count: Counter[int] = Counter()
    feature_taxids: set[int] = set()

    rows_processed = 0
    rows_skipped_missing_mapping = 0
    rows_skipped_empty = 0
    excluded_taxid_occurrences = 0
    excluded_taxid_mass = 0.0

    for _, seqid, pseudo_taxid_text, length_text, payload in iter_kraken_rows(args.database_tsv):
        rows_processed += 1

        seq_taxid = seqid_to_taxid.get(seqid)
        if seq_taxid is None:
            try:
                seq_taxid = int(pseudo_taxid_text)
            except ValueError:
                rows_skipped_missing_mapping += 1
                continue

        species_info = child_to_species.get(seq_taxid)
        if species_info is None:
            rows_skipped_missing_mapping += 1
            continue
        species_taxid, _ = species_info

        weight = float(length_text)
        row_feat_taxids, row_excl_occ, row_excl_mass = accumulate_weighted_taxid_counts(
            payload,
            weight,
            species_feature_weights[species_taxid],
            exclude_taxids=excluded_taxids,
        )
        excluded_taxid_occurrences += row_excl_occ
        excluded_taxid_mass        += row_excl_mass
        if not row_feat_taxids:
            rows_skipped_empty += 1
            continue

        species_total_weight[species_taxid]   += weight
        species_sequence_count[species_taxid] += 1
        feature_taxids.update(row_feat_taxids)

        if args.progress_every > 0 and rows_processed % args.progress_every == 0:
            report_progress(
                rows_processed=rows_processed,
                rows_skipped_missing_mapping=rows_skipped_missing_mapping,
                rows_skipped_empty=rows_skipped_empty,
                n_species=len(species_feature_weights),
                n_features=len(feature_taxids),
                start_time=start_time,
            )

    sorted_species_taxids = [
        taxid
        for taxid in sorted(species_feature_weights)
        if float(sum(species_feature_weights[taxid].values())) > 0.0
    ]
    sorted_feature_taxids = sorted(feature_taxids)
    species_index = {taxid: idx for idx, taxid in enumerate(sorted_species_taxids)}
    feature_index = {taxid: idx for idx, taxid in enumerate(sorted_feature_taxids)}

    rows_list: List[int] = []
    cols_list: List[int] = []
    data_list: List[float] = []
    metadata_rows: List[Dict] = []

    for species_taxid in sorted_species_taxids:
        raw_weights = species_feature_weights[species_taxid]
        total_weight = float(sum(raw_weights.values()))
        if total_weight <= 0:
            continue
        row_index = species_index[species_taxid]
        for feat_taxid, weight in raw_weights.items():
            rows_list.append(row_index)
            cols_list.append(feature_index[feat_taxid])
            data_list.append(weight / total_weight)
        metadata_rows.append({
            "species_index":          row_index,
            "species_taxid":          species_taxid,
            "species_name":           species_name_lookup.get(species_taxid, f"taxid_{species_taxid}"),
            "n_sequences":            int(species_sequence_count[species_taxid]),
            "total_reference_weight": float(species_total_weight[species_taxid]),
            "n_features":             len(raw_weights),
        })

    save_sparse_matrix(
        out_prefix.with_suffix(".matrix.npz"),
        np.asarray(rows_list, dtype=np.int32),
        np.asarray(cols_list, dtype=np.int32),
        np.asarray(data_list, dtype=np.float64),
        (len(sorted_species_taxids), len(sorted_feature_taxids)),
    )
    write_tsv(out_prefix.with_suffix(".species.tsv"), pd.DataFrame(metadata_rows))
    write_tsv(
        out_prefix.with_suffix(".features.tsv"),
        pd.DataFrame({
            "feature_index": np.arange(len(sorted_feature_taxids), dtype=np.int32),
            "taxid":         sorted_feature_taxids,
        }),
    )
    write_tsv(
        out_prefix.with_suffix(".summary.tsv"),
        pd.DataFrame([{
            "rows_processed":                        rows_processed,
            "rows_skipped_missing_mapping":          rows_skipped_missing_mapping,
            "rows_skipped_empty_after_filtering":    rows_skipped_empty,
            "n_species":                             len(sorted_species_taxids),
            "n_species_touched_before_filtering":    len(species_feature_weights),
            "n_features":                            len(sorted_feature_taxids),
            "exclude_taxid_count_requested":         int(len(user_excluded)),
            "exclude_taxid_count_applied":           int(len(excluded_taxids)),
            "exclude_taxids":                        ";".join(str(t) for t in sorted(excluded_taxids)),
            "excluded_taxid_occurrences":            int(excluded_taxid_occurrences),
            "excluded_taxid_mass":                   float(excluded_taxid_mass),
        }]),
    )
    report_progress(
        rows_processed=rows_processed,
        rows_skipped_missing_mapping=rows_skipped_missing_mapping,
        rows_skipped_empty=rows_skipped_empty,
        n_species=len(sorted_species_taxids),
        n_features=len(sorted_feature_taxids),
        start_time=start_time,
        final=True,
    )


if __name__ == "__main__":
    main()
