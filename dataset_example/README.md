# Example input files

This directory contains minimal mock input files illustrating the required format for
running `kraken_adna_kmer_screen`. They use placeholder paths and a small set of
well-known microbial taxa; they are **not** suitable for real analysis.

## Files

| File | Description |
|---|---|
| `units.tsv` | Sample/unit manifest: 2 samples × 2 sequencing lanes each |
| `species.tax_ids.tsv` | Taxonomy membership file — species level (20 entries) |
| `genus.tax_ids.tsv` | Taxonomy membership file — genus level (10 entries) |
| `target_genera.txt` | Example target genus filter (10 genus taxids) |

## Adapting for real data

**`units.tsv`** — replace placeholder paths with actual KrakenUniq classify and report
file paths. Each row is one sequencing unit (lane). Multiple units with the same
`sample_id` are aggregated automatically.

**Taxonomy files** — the real `species.tax_ids.tsv.gz` and `genus.tax_ids.tsv.gz` are
built alongside a KrakenUniq database and must cover all taxa present in the database.
They must be gzip-compressed for use with the workflow (the examples here are plain TSV
for readability). See the main README for details on building the reference database.

**`target_genera.txt`** — optional; restricts NNLS fitting to the listed genera.
Pre-built lists for bacteria, viruses, and eukaryotes are in `inputs/genera_*.txt`.
