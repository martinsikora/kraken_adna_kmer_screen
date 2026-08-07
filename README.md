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
  abundance + damage + coverage ──► aggregate_all ──► integrated summary table (.tsv.gz)

Per sample (auto):
  damage TSVs + integrated summary flags ──► plot_damage ──► damage_profile.pdf
                                            ──► damage_summary.pdf
                                            ──► damage_fractional.pdf
```

---

## Repository structure

```
kraken_adna_kmer_screen/
├── run_dataset.sh          # local runner
├── run_dataset_slurm.sh    # SLURM runner
├── workflow/
│   └── Snakefile
├── scripts/                # Python analysis scripts
├── inputs/                 # pre-built genus filter lists
└── dataset_example/        # template dataset directory
    └── config/
        ├── config.yaml     # workflow configuration
        └── units.tsv       # sample/unit manifest
```

Each dataset you analyse lives in its own directory (e.g. `cgg_2/014/`) with the same
`config/` layout as `dataset_example/`. The `workflow/`, `scripts/`, and `inputs/`
directories are shared across all datasets.

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
| R | ≥ 4.2 (plotting only) |
| R: ggplot2, ggrepel, readr, dplyr, scales, viridisLite | current |
| [KrakenUniq](https://github.com/fbreitwieser/krakenuniq) | ≥ 1.0 (for generating inputs) |

Install Python dependencies:

```bash
pip install snakemake numpy scipy pandas matplotlib
# or with conda:
conda install -c bioconda -c conda-forge snakemake numpy scipy pandas matplotlib
```

The damage plots are produced by R scripts (`scripts/plot_damage_*.R`); the rest
of the workflow is Python. Install the R side with:

```bash
conda install -c conda-forge r-base r-ggplot2 r-ggrepel r-readr r-dplyr r-scales r-viridislite
# or from within R:
install.packages(c("ggplot2", "ggrepel", "readr", "dplyr", "scales", "viridisLite"))
```

---

## Installation

```bash
git clone https://github.com/<org>/kraken_adna_kmer_screen.git
cd kraken_adna_kmer_screen
```

To use a shared deployment, symlink `workflow/`, `scripts/`, `inputs/`, and the runner
scripts into a shared directory so all datasets can share a single copy of the workflow:

```bash
DEPLOY=/path/to/shared/kraken_adna_kmer_screen
REPO=/path/to/kraken_adna_kmer_screen
mkdir -p "$DEPLOY/workflow"
ln -s "$REPO/workflow/Snakefile"       "$DEPLOY/workflow/Snakefile"
ln -s "$REPO/scripts"                  "$DEPLOY/scripts"
ln -s "$REPO/inputs"                   "$DEPLOY/inputs"
ln -s "$REPO/run_dataset.sh"           "$DEPLOY/run_dataset.sh"
ln -s "$REPO/run_dataset_slurm.sh"     "$DEPLOY/run_dataset_slurm.sh"
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

Set `reference_dir` in your dataset's `config/config.yaml` to the directory containing
these files.

### Taxonomy membership files

The workflow also requires pre-built taxonomy membership files:

- `species.tax_ids.tsv.gz` — maps every descendant taxid to a species entry
- `genus.tax_ids.tsv.gz` — maps every descendant taxid to a genus entry

These are TSV files (gzip-compressed) with columns:

```
tax_rank    tax_id    tax_name           tax_ids_descendant
species     632       Yersinia pestis    632
species     632       Yersinia pestis    214092
species     632       Yersinia pestis    349746
...
```

Each descendant taxid occupies its own row; a species/genus entry is repeated once per
descendant strain or assembly taxid.

They are typically generated alongside the KrakenUniq database from the NCBI taxonomy
dump using the companion `build_taxlists` utility (not included here). See
`dataset_example/species.tax_ids.tsv` and `dataset_example/genus.tax_ids.tsv` for the
expected format.

---

## Preparing inputs

### 1. Sample/unit manifest (`units.tsv`)

A tab-separated file listing every sequencing unit (lane) to process.
Copy `dataset_example/config/units.tsv` as a starting point.

| Column | Description |
|---|---|
| `unit_id` | Globally unique identifier for this sequencing unit |
| `sample_id` | Sample identifier; units with the same `sample_id` are merged |
| `kraken_class` | Path to gzip-compressed KrakenUniq classify output (`.tsv.gz`) |
| `kraken_report` | Path to KrakenUniq report TSV (uncompressed) |

`unit_id` values must be unique across the entire manifest. The Snakefile raises an
error on startup if duplicates are detected.

### 2. Taxonomy files

Set the absolute paths to `species.tax_ids.tsv.gz` and `genus.tax_ids.tsv.gz` in
`config/config.yaml` (`species_taxids` and `genus_taxids` keys).

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
all genera is in `inputs/genera_all.txt`. Reference these with absolute paths in
`config/config.yaml`.

Set `target_genus_file` in `config/config.yaml` to the desired list path, or leave it
empty (`""`) to fit all genera present in the reference.

---

## Configuration

Copy `dataset_example/config/config.yaml` into your dataset directory and edit it.
Key parameters:

### Paths

| Parameter | Description |
|---|---|
| `reference_dir` | Directory with pre-built reference matrix files |
| `species_taxids` | Path to `species.tax_ids.tsv.gz` |
| `genus_taxids` | Path to `genus.tax_ids.tsv.gz` |
| `units_tsv` | Path to sample/unit manifest |
| `target_genus_file` | Optional genus filter file (empty = disabled) |
| `exclude_taxids` | Taxids excluded during vectorization (defaults: `0,1,2,131567`) |

### NNLS fitting

| Parameter | Default | Description |
|---|---|---|
| `fit_mode` | `fast` | `fast` (two-pass candidate pruning) or `exact` |
| `fit_granularity` | `genus` | `species`, `genus`, or `within-genus` |
| `fit_constraint` | `nnls` | `nnls` or `simplex` |
| `max_candidates` | `1024` | Max candidate species per genus (fast mode) |
| `max_feature_support` | `0` | Drop features present in more than this many species (`0` disables filter) |
| `min_genus_relative_abundance` | `0.0` | Minimum genus abundance before within-genus fit |
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
| `damage_plot_required_hit_flags` | `["damage_pvalue","within_genus_relative_abundance","classified_rate"]` | Required hit tokens for selecting taxa in damage plots |
| `damage_plot_max_keys` | `200` | Max taxa/pages per damage plot PDF |

### Coverage evenness

| Parameter | Default | Description |
|---|---|---|
| `evenness_min_reads` | `50` | Minimum reads for evenness reporting |
| `evenness_ranks` | `[species,genus]` | Taxonomic ranks emitted in coverage table |

### Hit criteria thresholds

The integrated summary table (`all_samples.summary.tsv.gz`) stores passing criteria
as semicolon-delimited tokens in `hit_criteria_flag`:

| Parameter | Default | Criterion |
|---|---|---|
| `hit_max_damage_pvalue` | `0.05` | `damage_pvalue` < threshold |
| `hit_min_within_genus_ra` | `0.1` | `within_genus_relative_abundance` ≥ threshold |
| `hit_min_classified_rate` | `0.5` | `plateau_classified_rate` ≥ threshold |

The `evenness_index` criterion is depth-aware. `evenness_index` is the
Lander-Waterman ratio `E = cov / (1 − exp(−dup·cov))`, which degenerates at both
ends of the depth range: when mean genome depth `λ = dup·cov` is far below 1,
`E ≈ 1/dup`; when it is far above 1, `E ≈ cov`. Screening data sits almost
entirely in the first regime, so a single threshold on `E` tests k-mer
duplication for shallow taxa but coverage breadth for deep ones. The criterion
therefore applies the test appropriate to each regime:

| Parameter | Default | Criterion |
|---|---|---|
| `hit_evenness_mode` | `depth-aware` | `depth-aware` or `legacy` |
| `hit_evenness_lambda_split` | `0.1` | `λ = dup·cov` boundary between regimes |
| `hit_max_dup_shallow` | `3.0` | shallow (`λ` < split): `dup` < threshold |
| `hit_min_cov_deep` | `0.05` | deep (`λ` ≥ split): `cov` > threshold |
| `hit_min_evenness` | `0.5` | `legacy` mode, and fallback when `dup`/`cov` are absent |

Set `hit_evenness_mode: legacy` to restore the previous single-threshold
behaviour (`evenness_index` > `hit_min_evenness`).

When all four criteria pass, `hit_criteria_flag` is:
`damage_pvalue;evenness_index;within_genus_relative_abundance;classified_rate`.

Rows are included in `all_samples.summary.tsv.gz` only when abundance, evenness,
damage, and classified-rate statistics are all available for that sample/species.

### Compute resources

| Parameter | Description |
|---|---|
| `resources.screen_unit` | Memory/runtime for per-unit vectorization |
| `resources.aggregate_sample` | Memory/runtime for per-sample aggregation |
| `resources.fit_abundance` | Memory/runtime for NNLS fitting |
| `resources.coverage_evenness` | Memory/runtime for evenness parsing |
| `resources.aggregate_all` | Memory/runtime for cross-sample summary |
| `resources.plot_damage` | Memory/runtime for per-sample PDF plotting |

---

## Running the workflow

Run from the workflow root directory (where `run_dataset.sh` lives), passing the path
to your dataset directory as the first argument.

```bash
# Local run
bash run_dataset.sh my_dataset

# Dry-run to preview jobs
bash run_dataset.sh my_dataset --dry-run

# Pass extra Snakemake options after --
bash run_dataset.sh my_dataset -- --cores 32 --rerun-incomplete

# SLURM submission
bash run_dataset_slurm.sh my_dataset
bash run_dataset_slurm.sh my_dataset --jobs 50 --partition highmem
bash run_dataset_slurm.sh my_dataset --dry-run
bash run_dataset_slurm.sh my_dataset --unlock   # after a failed run
```

Use `-h` / `--help` for the full option list of either runner script.

The integrated summary table and per-sample damage PDFs are built by the default `all` target.

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
| `all_samples.summary.tsv.gz` | Single integrated sample-species table with abundance, damage, coverage, and hit-criterion tokens |

### Key output columns

**`all_samples.summary.tsv.gz`** (selected columns):

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
| `hit_criteria_flag` | Semicolon-delimited passing criteria tokens from: `damage_pvalue`, `evenness_index`, `within_genus_relative_abundance`, `classified_rate` |

---

## Damage visualisation

Three PDFs are produced per sample:

- **`damage_profile.pdf`** — Absolute-position profiles (0 = terminal k-mer) for the
  5′ and 3′ ends of each hit species. Shaded plateau region and per-end damage scores
  are annotated. One page per species.

- **`damage_summary.pdf`** — Scatter plot of all profiled taxa. X axis: baseline
  classified k-mer rate; Y axis: damage score (%). Point size = log₁₀(reads);
  colour = `evenness_index` on a fixed 0–1 scale (grey where unavailable).
  Triangles = significant. Only selected hit species are annotated.
  Set `damage_plot_color_by: pvalue` to colour by −log₁₀(p-value) instead.
  Colouring by evenness adds information the plot does not already carry — both
  axes and the marker shape are damage-derived, whereas evenness reports whether
  coverage is genome-wide or clumped.

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

## License

MIT — see [LICENSE](LICENSE).
