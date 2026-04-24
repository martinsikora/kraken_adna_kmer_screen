#!/usr/bin/env python3
"""
Generate kraken_adna_kmer dataset configs from krakenuniq datasets.

Behavior:
- Discover datasets under {source_root}/cgg_1/* and {source_root}/cgg_2/*
- Copy template config.yaml into {target_root}/{dataset}/config/config.yaml
- Build units.tsv with columns:
    unit_id, sample_id, kraken_class, kraken_report
- Map sample_id from metadata using:
    1) exact: row.unit_id -> metadata.unit_id
    2) fallback: row.unit_id -> metadata.unit_prefix
- Exclude rows with missing/ambiguous mappings or missing classify/report paths.
- Write per-dataset warnings and global setup summary.
"""

from __future__ import annotations

import argparse
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


DEFAULT_TEMPLATE_CONFIG = (
    "/datasets/apollo/fastq_screening/kraken_adna_kmer/cgg_2/014/config/config.yaml"
)
DEFAULT_SOURCE_ROOT = "/datasets/apollo/fastq_screening/krakenuniq"
DEFAULT_TARGET_ROOT = "/datasets/apollo/fastq_screening/kraken_adna_kmer"
DEFAULT_METADATA = (
    "/datasets/apollo/databases/metadata/"
    "lundbeck_fastq_inventory_2026-04-10_extended_curated.tsv"
)


@dataclass
class MapResult:
    sample_id: str | None
    method: str | None
    reason: str | None
    detail: str


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--template-config", default=DEFAULT_TEMPLATE_CONFIG)
    p.add_argument("--source-root", default=DEFAULT_SOURCE_ROOT)
    p.add_argument("--target-root", default=DEFAULT_TARGET_ROOT)
    p.add_argument("--metadata", default=DEFAULT_METADATA)
    p.add_argument(
        "--existing-policy",
        choices=["skip", "overwrite", "fail"],
        default="skip",
        help="How to handle datasets where target config/ already exists.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan and report actions without writing files.",
    )
    return p.parse_args()


def read_prefix_from_krakenuniq_config(config_yml: Path) -> str:
    """
    Parse "prefix" from simple YAML shape used by krakenuniq configs:
      prefix:
        hum_microbe_20250624
    or:
      prefix: hum_microbe_20250624
    """
    lines = config_yml.read_text(encoding="utf-8").splitlines()
    for i, raw in enumerate(lines):
        line = raw.strip()
        if not line.startswith("prefix:"):
            continue

        inline = line.split(":", 1)[1].strip()
        if inline:
            return inline.strip("\"'")

        for j in range(i + 1, len(lines)):
            nxt = lines[j].strip()
            if not nxt or nxt.startswith("#"):
                continue
            # reached next top-level-like key before value
            if re.match(r"^[A-Za-z0-9_][A-Za-z0-9_-]*\s*:\s*$", nxt):
                break
            if ":" in nxt:
                break
            return nxt.strip("\"'")

    raise ValueError(f"Could not parse prefix from {config_yml}")


def make_unique_map(df: pd.DataFrame, key_col: str, value_col: str) -> tuple[dict[str, str], dict[str, list[str]]]:
    grouped = (
        df[[key_col, value_col]]
        .dropna(subset=[key_col, value_col])
        .drop_duplicates()
        .groupby(key_col)[value_col]
        .agg(lambda s: sorted(set(s)))
    )
    unique = {k: vals[0] for k, vals in grouped.items() if len(vals) == 1}
    ambiguous = {k: vals for k, vals in grouped.items() if len(vals) > 1}
    return unique, ambiguous


def stem_from_fq_path(fq_path: str) -> str:
    name = Path(fq_path).name
    for suffix in (".fastp.coll.fq.gz", ".fastp.coll.fastq.gz", ".fq.gz", ".fastq.gz"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    if name.endswith(".gz"):
        return name[:-3]
    return name


def resolve_sample_id(
    unit_id: str,
    uid_unique: dict[str, str],
    uid_ambiguous: dict[str, list[str]],
    up_unique: dict[str, str],
    up_ambiguous: dict[str, list[str]],
) -> MapResult:
    if unit_id in uid_unique:
        return MapResult(uid_unique[unit_id], "unit_id_exact", None, "")
    if unit_id in uid_ambiguous:
        vals = ",".join(uid_ambiguous[unit_id])
        return MapResult(None, None, "ambiguous_mapping", f"metadata.unit_id maps to multiple sample_id values: {vals}")
    if unit_id in up_unique:
        return MapResult(up_unique[unit_id], "unit_id_to_metadata_unit_prefix", None, "")
    if unit_id in up_ambiguous:
        vals = ",".join(up_ambiguous[unit_id])
        return MapResult(
            None,
            None,
            "ambiguous_mapping",
            f"fallback metadata.unit_prefix maps to multiple sample_id values: {vals}",
        )
    return MapResult(None, None, "missing_mapping", "no metadata match by unit_id or fallback unit_prefix")


def resolve_paths(source_dataset_dir: Path, row: pd.Series, prefix: str) -> tuple[str | None, str | None, list[dict[str, str]]]:
    stem = stem_from_fq_path(str(row["fq"]))
    warnings: list[dict[str, str]] = []

    candidates = []
    for key in ("unit_prefix", "unit_id"):
        val = str(row.get(key, "") or "").strip()
        if val and val not in candidates:
            candidates.append(val)

    probe_info: list[tuple[str, Path, Path, bool, bool]] = []
    for candidate in candidates:
        base = source_dataset_dir / "results" / candidate
        classify = base / "classify" / f"{stem}.{prefix}.krakenuniq_class.tsv.gz"
        report = base / "report" / f"{stem}.{prefix}.krakenuniq_report.tsv"
        class_ok = classify.exists()
        report_ok = report.exists()
        probe_info.append((candidate, classify, report, class_ok, report_ok))
        if class_ok and report_ok:
            return str(classify), str(report), warnings

    for candidate, classify, report, class_ok, report_ok in probe_info:
        if not class_ok:
            warnings.append(
                {
                    "reason": "missing_classify",
                    "detail": f"missing classify for candidate '{candidate}': {classify}",
                }
            )
        if not report_ok:
            warnings.append(
                {
                    "reason": "missing_report",
                    "detail": f"missing report for candidate '{candidate}': {report}",
                }
            )
    return None, None, warnings


def discover_source_datasets(source_root: Path) -> list[Path]:
    out = []
    for cohort in ("cgg_1", "cgg_2"):
        base = source_root / cohort
        if not base.exists():
            continue
        for dataset_dir in sorted(p for p in base.iterdir() if p.is_dir()):
            units = dataset_dir / "config" / "units.tsv"
            cfg = dataset_dir / "config" / "config.yml"
            if units.exists() and cfg.exists():
                out.append(dataset_dir)
    return out


def ensure_parent(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()

    template_config = Path(args.template_config)
    source_root = Path(args.source_root)
    target_root = Path(args.target_root)
    metadata_path = Path(args.metadata)

    if not template_config.exists():
        raise FileNotFoundError(f"Template config not found: {template_config}")
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    metadata = pd.read_csv(
        metadata_path,
        sep="\t",
        dtype=str,
        usecols=["unit_id", "unit_prefix", "sample_id"],
    ).dropna(subset=["sample_id"])
    uid_unique, uid_ambiguous = make_unique_map(metadata, "unit_id", "sample_id")
    up_unique, up_ambiguous = make_unique_map(metadata, "unit_prefix", "sample_id")

    datasets = discover_source_datasets(source_root)
    if not datasets:
        raise RuntimeError(f"No datasets found under {source_root}")

    summary_rows: list[dict[str, object]] = []
    global_warnings: list[dict[str, str]] = []

    for source_dataset_dir in datasets:
        cohort = source_dataset_dir.parent.name
        dataset_id = source_dataset_dir.name
        dataset_key = f"{cohort}/{dataset_id}"
        target_dataset_dir = target_root / cohort / dataset_id
        target_config_dir = target_dataset_dir / "config"

        units_path = source_dataset_dir / "config" / "units.tsv"
        source_config_yml = source_dataset_dir / "config" / "config.yml"
        kraken_prefix = read_prefix_from_krakenuniq_config(source_config_yml)

        if target_config_dir.exists():
            if args.existing_policy == "fail":
                raise RuntimeError(f"Target config exists (policy=fail): {target_config_dir}")
            if args.existing_policy == "skip":
                global_warnings.append(
                    {
                        "dataset": dataset_key,
                        "unit_id": "",
                        "unit_prefix": "",
                        "sample_id": "",
                        "reason": "skipped_existing_config",
                        "detail": f"skipped existing config dir: {target_config_dir}",
                    }
                )
                summary_rows.append(
                    {
                        "dataset": dataset_key,
                        "status": "skipped_existing_config",
                        "units_in": 0,
                        "units_written": 0,
                        "warnings": 1,
                        "missing_mapping": 0,
                        "ambiguous_mapping": 0,
                        "missing_classify": 0,
                        "missing_report": 0,
                    }
                )
                continue

        units_df = pd.read_csv(units_path, sep="\t", dtype=str).fillna("")
        out_rows: list[dict[str, str]] = []
        warn_rows: list[dict[str, str]] = []
        warn_counts = {
            "missing_mapping": 0,
            "ambiguous_mapping": 0,
            "missing_classify": 0,
            "missing_report": 0,
        }

        for row in units_df.itertuples(index=False):
            unit_id = str(getattr(row, "unit_id", "") or "")
            unit_prefix = str(getattr(row, "unit_prefix", "") or "")
            fq = str(getattr(row, "fq", "") or "")
            row_series = pd.Series({"unit_id": unit_id, "unit_prefix": unit_prefix, "fq": fq})

            mapped = resolve_sample_id(unit_id, uid_unique, uid_ambiguous, up_unique, up_ambiguous)
            if mapped.sample_id is None:
                warn_counts[mapped.reason or "missing_mapping"] += 1
                warn_rows.append(
                    {
                        "dataset": dataset_key,
                        "unit_id": unit_id,
                        "unit_prefix": unit_prefix,
                        "sample_id": "",
                        "reason": mapped.reason or "missing_mapping",
                        "detail": mapped.detail,
                    }
                )
                continue

            kraken_class, kraken_report, path_warnings = resolve_paths(source_dataset_dir, row_series, kraken_prefix)
            if kraken_class is None or kraken_report is None:
                for w in path_warnings:
                    warn_counts[w["reason"]] += 1
                    warn_rows.append(
                        {
                            "dataset": dataset_key,
                            "unit_id": unit_id,
                            "unit_prefix": unit_prefix,
                            "sample_id": mapped.sample_id,
                            "reason": w["reason"],
                            "detail": w["detail"],
                        }
                    )
                continue

            out_rows.append(
                {
                    "unit_id": unit_id,
                    "sample_id": mapped.sample_id,
                    "kraken_class": kraken_class,
                    "kraken_report": kraken_report,
                }
            )

        out_df = pd.DataFrame(out_rows, columns=["unit_id", "sample_id", "kraken_class", "kraken_report"])
        if not out_df.empty:
            out_df = out_df.sort_values(["sample_id", "unit_id"], kind="stable").reset_index(drop=True)

        warn_df = pd.DataFrame(
            warn_rows,
            columns=["dataset", "unit_id", "unit_prefix", "sample_id", "reason", "detail"],
        )
        global_warnings.extend(warn_rows)

        if not args.dry_run:
            target_config_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(template_config, target_config_dir / "config.yaml")
            out_df.to_csv(target_config_dir / "units.tsv", sep="\t", index=False)
            warn_df.to_csv(target_config_dir / "setup_warnings.tsv", sep="\t", index=False)

        status = "created" if not args.dry_run else "planned"
        summary_rows.append(
            {
                "dataset": dataset_key,
                "status": status,
                "units_in": int(len(units_df)),
                "units_written": int(len(out_df)),
                "warnings": int(len(warn_df)),
                "missing_mapping": int(warn_counts["missing_mapping"]),
                "ambiguous_mapping": int(warn_counts["ambiguous_mapping"]),
                "missing_classify": int(warn_counts["missing_classify"]),
                "missing_report": int(warn_counts["missing_report"]),
            }
        )

    summary_df = pd.DataFrame(
        summary_rows,
        columns=[
            "dataset",
            "status",
            "units_in",
            "units_written",
            "warnings",
            "missing_mapping",
            "ambiguous_mapping",
            "missing_classify",
            "missing_report",
        ],
    ).sort_values("dataset", kind="stable")

    global_warn_df = pd.DataFrame(
        global_warnings,
        columns=["dataset", "unit_id", "unit_prefix", "sample_id", "reason", "detail"],
    ).sort_values(["dataset", "reason", "unit_id"], kind="stable")

    summary_path = target_root / "setup_config_summary.tsv"
    warnings_path = target_root / "setup_config_warnings.tsv"
    if not args.dry_run:
        ensure_parent(summary_path, dry_run=False)
        summary_df.to_csv(summary_path, sep="\t", index=False)
        global_warn_df.to_csv(warnings_path, sep="\t", index=False)

    print(
        f"[setup_adna_dataset_configs] datasets={len(summary_df)} "
        f"planned_or_created={int((summary_df['status'] != 'skipped_existing_config').sum())} "
        f"skipped_existing={int((summary_df['status'] == 'skipped_existing_config').sum())} "
        f"warnings={len(global_warn_df)} "
        f"dry_run={args.dry_run}"
    )
    print(f"[setup_adna_dataset_configs] summary_path={summary_path}")
    print(f"[setup_adna_dataset_configs] warnings_path={warnings_path}")


if __name__ == "__main__":
    main()

