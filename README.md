# kraken_adna_kmer_screen

A Snakemake workflow for screening ancient DNA (aDNA) samples for pathogen and species
presence using [KrakenUniq](https://github.com/fbreitwieser/krakenuniq) classification
output. For each sample the workflow jointly estimates:

1. **Species/genus relative abundances** via non-negative least squares (NNLS) fitting
   against a pre-built k-mer fingerprint reference matrix.
2. **aDNA damage profiles** by tracking the fraction of unclassified k-mers at read
   termini, stratified by read length.
3. **Coverage evenness** (Lander-Waterman index) from KrakenUniq k-mer coverage statistics.

Both the abundance vector and the damage accumulator are built in a **single streaming
pass** through each classify file, so large (multi-GB) inputs are processed efficiently.

---

## Workflow overview

```
For each sequencing unit (lane):
  krakenuniq_class.tsv.gz ──► screen_unit ──► vector.npz + damage_arrays.npz

For each sample (merge lanes):
  vector.npz × N ──► aggregate_sample ──► merged vector + damage TSVs
  krakenuniq_report.tsv × N ──► coverage_evenness ──► coverage.tsv

For each sample:
  merged vector + reference matrix ──► fit_abundance ──► abundance.tsv

All samples:
  abundance + damage + coverage ──► aggregate_all ──► summary tables + hit table

Per sample (auto):
  damage TSVs + hit table ──► plot_damage ──► damage_profile.pdf
                                            ──► damage_summary.pdf
                                            ──► damage_fractional.pdf
```

---

## Requirements

| Dependency | Tested version |
|---|---|
| Python | ≥ 3.9 |
| [Snakemake](https://snakemake.readthedocs.io/) | ≥ 7.0 |
| numpy | ≥ 1.23 |
| scipy | ≥ 1.9 |
| pandas | ≥ 1.5 |
| matplotlib | ≥ 3.6 |
| [KrakenUniq](https://github.com/fbreitwieser/krakenuniq) | ≥ 1.0 (for generating inputs) |

Install Python dependencies:

```bash
pip install snakemake numpy scipy pandas matplotlib
# or with conda:
conda install -c bioconda -c conda-forge snakemake numpy scipy pandas matplotlib
```

---

## Installation

```bash
git clone https://github.com/<org>/kraken_adna_kmer_screen.git
cd kraken_adna_kmer_screen
```

---

## Database setup (run once per KrakenUniq database)

Before running the workflow you need to build a species-level reference matrix from your
KrakenUniq database. This step is separate from the per-sample workflow and only needs to
be repeated when the database changes.

```bash
python scripts/build_reference_matrix.py \
    --db-tsv        /path/to/krakendb/database.kraken.tsv \
    --seqid-map     /path/to/krakendb/seqid2taxid.map \
    --species-taxids inputs/species.tax_ids.tsv.gz \
    --out-prefix    /path/to/krakendb/kraken_adna_kmer_screen_db/reference \
    --exclude-taxids 0 1 2 131567
```

This writes five files to the `--out-prefix` directory:

| File | Description |
|---|---|
| `reference.matrix.npz` | Sparse CSR matrix (species × features) |
| `reference.matrix.csc.npz` | CSC sidecar for efficient column access |
| `reference.species.tsv` | Row metadata (species index, taxid, name) |
| `reference.features.tsv` | Column metadata (feature index, taxid) |
| `reference.summary.tsv` | Build diagnostics |

Set `reference_dir` in `config/config.yaml` to the directory containing these files.

### Taxonomy membership files

The workflow also requires pre-built taxonomy membership files:

- `species.tax_ids.tsv.gz` — maps every descendant taxid to a species entry
- `genus.tax_ids.tsv.gz` — maps every descendant taxid to a genus entry

These are TSV files (gzip-compressed) with columns:

```
tax_rank    tax_id    tax_name    tax_ids_descendant
species     632       Yersinia pestis    632,214092,349746
...
```

They are typically generated alongside the KrakenUniq database from the NCBI taxonomy
dump using the companion `build_taxlists` utility (not included here). See
`example/species.tax_ids.tsv` and `example/genus.tax_ids.tsv` for the expected format.

---

## Preparing inputs

### 1. Sample/unit manifest (`units.tsv`)

A tab-separated file listing every sequencing unit (lane) to process.
Copy `example/units.tsv` as a starting point.

| Column | Description |
|---|---|
| `unit_id` | Globally unique identifier for this sequencing unit |
| `sample_id` | Sample identifier; units with the same `sample_id` are merged |
| `kraken_class` | Path to gzip-compressed KrakenUniq classify output (`.tsv.gz`) |
| `kraken_report` | Path to KrakenUniq report TSV (uncompressed) |

`unit_id` values must be unique across the entire manifest. The Snakefile raises an
error on startup if duplicates are detected.

### 2. Taxonomy files

Place `species.tax_ids.tsv.gz` and `genus.tax_ids.tsv.gz` in the `inputs/` directory
(or update the paths in `config/config.yaml`).

### 3. Target genus filter (optional)

To restrict abundance fitting to a specific set of genera, provide a plain-text file
with one genus name or taxid per line (lines starting with `#` are ignored):

```
# Bacterial pathogens only
629
1763
234
```

Pre-built lists covering bacteria (`genera_bacteria.txt`), viruses (`genera_virus.txt`),
and eukaryotes (`genera_eukaryota.txt`) are included in `inputs/`. A combined list of
all genera is in `inputs/genera_all.txt`.

Set `target_genus_file` in `config/config.yaml` to the desired list path, or leave it
empty (`""`) to fit all genera present in the reference.

---

## Configuration

Edit `config/config.yaml` before running. Key parameters:

### Paths

| Parameter | Description |
|---|---|
| `reference_dir` | Directory with pre-built reference matrix files |
| `species_taxids` | Path to `species.tax_ids.tsv.gz` |
| `genus_taxids` | Path to `genus.tax_ids.tsv.gz` |
| `units_tsv` | Path to sample/unit manifest |
| `target_genus_file` | Optional genus filter file (empty = disabled) |

### NNLS fitting

| Parameter | Default | Description |
|---|---|---|
| `fit_mode` | `fast` | `fast` (two-pass candidate pruning) or `exact` |
| `fit_granularity` | `genus` | `species`, `genus`, or `within-genus` |
| `fit_constraint` | `nnls` | `nnls` or `simplex` |
| `max_candidates` | `1024` | Max candidate species per genus (fast mode) |
| `target_genus` | `[]` | Inline list of genus names/taxids to fit |
| `restrict_to_target_genus_features` | `false` | Drop features outside target genera |

### aDNA damage

| Parameter | Default | Description |
|---|---|---|
| `damage_max_pos` | `25` | Positions from read end to profile |
| `damage_min_reads` | `100` | Minimum reads to compute damage statistics |
| `damage_n_bins` | `50` | Bins for fractional-position profiles |
| `damage_adaptive_plateau` | `true` | Auto-detect plateau window per taxon |
| `damage_plateau_search_start` | `3` | Plateau search window start (positions from end) |
| `damage_plateau_search_end` | `10` | Plateau search window end |
| `damage_strata` | `["31-40","41-55","56-75","76-100"]` | Read-length strata (bp) |
| `damage_annotate_min_classified` | `0.6` | Min baseline classified rate for plot annotation |

### Hit table thresholds

A species enters the final hit table (`all_samples.hits.tsv`) only when all three
criteria are met:

| Parameter | Default | Criterion |
|---|---|---|
| `hit_max_damage_pvalue` | `0.05` | `damage_pvalue` < threshold |
| `hit_min_evenness` | `0.5` | `evenness_index` > threshold |
| `hit_min_within_genus_ra` | `0.1` | `within_genus_relative_abundance` ≥ threshold |

---

## Running the workflow

```bash
# Local run (all samples + damage plots)
snakemake --cores 32 --resources mem_mb=64000

# Dry-run to preview jobs without executing
snakemake -n

# Resume after a partial run
snakemake --cores 32 --rerun-incomplete

# Cluster submission (SLURM example)
snakemake --cores 200 \
    --executor slurm \
    --default-resources slurm_account=<account> slurm_partition=<partition>
```

All six summary tables and per-sample damage PDFs are built by the default `all` target.

---

## Outputs

### Per-sample files (`results/samples/{sample_id}/`)

| File | Description |
|---|---|
| `{sample_id}.abundance.tsv` | Per-species NNLS results and relative abundances |
| `{sample_id}.genus.tsv` | Per-genus aggregated abundances |
| `{sample_id}.fit.tsv` | NNLS fit diagnostics |
| `{sample_id}.damage_global.tsv` | Per-species damage scores (summary) |
| `{sample_id}.damage_stats.tsv` | Per-species per-end damage statistics and plateau estimates |
| `{sample_id}.damage_profile.tsv` | Per-species absolute-position damage profiles |
| `{sample_id}.damage_fractional_profile.tsv` | Per-species fractional-position profiles by read-length stratum |
| `{sample_id}.coverage.tsv` | Per-taxon coverage and evenness statistics |
| `{sample_id}.damage_profile.pdf` | Absolute-position damage plots (hit species) |
| `{sample_id}.damage_summary.pdf` | Damage biplot across all profiled taxa |
| `{sample_id}.damage_fractional.pdf` | Fractional-position plots by read-length stratum (hit species) |

### Workflow-level summary (`results/summary/`)

| File | Description |
|---|---|
| `all_samples.abundance.tsv` | Stacked per-species abundance table (all samples) |
| `all_samples.damage.tsv` | Stacked per-species damage scores (all samples) |
| `all_samples.coverage.tsv` | Stacked per-taxon coverage statistics (all samples) |
| `all_samples.species.tsv` | Outer join of abundance + damage (all samples, all species) |
| `all_samples.hits.tsv` | Final hit table — species passing all three filters |
| `all_samples.summary.tsv` | One row per sample with top species, total reads, fit metrics |

### Key output columns

**`all_samples.hits.tsv`** (selected columns):

| Column | Description |
|---|---|
| `sample_id` | Sample identifier |
| `species_name` | Species name |
| `relative_abundance` | NNLS-estimated global relative abundance (0–1) |
| `within_genus_relative_abundance` | Relative abundance within genus |
| `rank` | Global abundance rank (1 = most abundant) |
| `n_reads` | Reads classified to this species |
| `damage_score` | Fraction of unclassified terminal k-mers above plateau (5′ end) |
| `damage_pvalue` | One-sided p-value for damage excess above plateau |
| `plateau_classified_rate` | Baseline k-mer classified rate (1 − plateau_frac_unclassified) |
| `evenness_index` | Lander-Waterman evenness (1 = Poisson-uniform, <1 = clumped) |
| `cov` | Breadth of k-mer coverage (fraction of genome represented) |
| `dup` | Mean k-mer depth (total k-mers / unique k-mers) |

---

## Damage visualisation

Three PDFs are produced per sample:

- **`damage_profile.pdf`** — Absolute-position profiles (0 = terminal k-mer) for the
  5′ and 3′ ends of each hit species. Shaded plateau region and per-end damage scores
  are annotated. One page per species.

- **`damage_summary.pdf`** — Scatter plot of all profiled taxa. X axis: baseline
  classified k-mer rate; Y axis: damage score (%). Point size = log₁₀(reads); colour
  = −log₁₀(p-value). Triangles = significant. Hit species are always annotated;
  other significant taxa with classified rate ≥ `damage_annotate_min_classified` are
  annotated with an asterisk.

- **`damage_fractional.pdf`** — Fractional-position profiles (0 = 5′, 1 = 3′) for
  hit species, stratified by read-length. Shows the U-shaped overlap artifact in
  short reads and the flat interior plateau in longer reads. One page per species.

---

## Evenness index

The coverage evenness index is the Lander-Waterman ratio:

```
E = cov / (1 − exp(−dup × cov))
```

where `cov` is the fraction of the reference genome covered by at least one unique
k-mer (breadth), and `dup` is the mean depth of covered positions. `E ≈ 1` indicates
coverage consistent with uniform (Poisson) read placement; `E < 1` indicates reads are
clumped relative to expectation.

---

## Citation

If you use this workflow, please cite:

> [manuscript in preparation]

---

## License

MIT — see [LICENSE](LICENSE).
