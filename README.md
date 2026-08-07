# kraken_adna_kmer_screen

A Snakemake workflow for screening ancient DNA (aDNA) samples for pathogen and species
presence using [KrakenUniq](https://github.com/fbreitwieser/krakenuniq) classification
output. For each sample the workflow jointly estimates:

1. **Species/genus relative abundances** via non-negative least squares (NNLS) fitting
   against a pre-built k-mer fingerprint reference matrix.
2. **aDNA damage profiles** by tracking the fraction of unclassified k-mers at read
   termini, both pooled and split by read-length stratum.
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

Per sample:
  abundance + damage + coverage ──► summarize_sample ──► summary.tsv.gz (hit flags)

Per sample (auto):
  damage TSVs + that sample's hit flags ──► plot_damage ──► damage_profile.pdf
                                            ──► damage_summary.pdf

All samples:
  abundance + damage + coverage ──► aggregate_all ──► integrated summary table (.tsv.gz)
```

Every column of the summary, `hit_criteria_flag` included, is a per-row function
of a single sample's own evidence — the merges are on `(sample_id,
species_name)`, the criteria compare a row against fixed config thresholds, and
the sort is within sample. So a sample's summary and its damage plots are built
as soon as that sample finishes, without waiting for the rest of the dataset.
`aggregate_all` remains a genuine cross-sample step and still needs them all.

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
| `damage_min_read_length` | `30` | Reads shorter than this are excluded from damage estimation |
| `damage_max_read_length` | `75` | Reads longer than this are excluded (see below) |
| `damage_max_pos` | `10` | Positions from read end to profile |
| `damage_min_reads` | `100` | Minimum reads to compute damage statistics |
| `damage_adaptive_plateau` | `true` | Auto-detect plateau window per taxon |
| `damage_plateau_search_start` | `3` | Plateau search window start (positions from end) |
| `damage_plateau_search_end` | `9` | Plateau search window end (must be < `damage_max_pos`) |
| `damage_strata` | `["30-55","56-75"]` | Read-length strata for the damage profile, inside the damage length window |
| `damage_plot_required_hit_flags` | `["damage_pvalue","within_genus_relative_abundance","classified_rate"]` | Required hit tokens for selecting taxa in damage plots |
| `damage_plot_max_keys` | `200` | Max taxa/pages per damage plot PDF |

#### Read-length window for damage

Damage is estimated only from reads within
`damage_min_read_length` … `damage_max_read_length`. Abundance, coverage and
evenness use all reads; only the damage accumulators are gated.

**Upper bound.** It must sit below the shortest sequencing read length used for
the sample. A read at the read-length cap is a truncated molecule: its 3′ end is
a sequencing cut-off rather than a molecule terminus, so it carries no terminal
damage and dilutes the 3′ estimate. On a 100 bp run the 3′ terminal excess falls
from ~+4.4 percentage points for reads under 95 bp to +0.3 at 98–99 bp, so the
contamination begins a couple of bases below the cap rather than at it. Pooling
lanes of different read length also mixes different cap positions into a single
profile, which the fixed window avoids.

**Lower bound.** With k-mer size *k*, a read reaches k-mer position *j* only if
its length is at least *k + j*. Positions beyond that are computed from
progressively fewer — and longer — reads, and longer reads carry systematically
higher unclassified rates, which biases the plateau upward. Keep
`damage_max_pos` small enough that most reads in the window reach the last
profiled position: at the defaults (`30`–`75`, k=29) roughly 84% of reads reach
position 10.

Narrowing the window costs reads: at the defaults about 57% of a typical
library is retained for damage. Lower `damage_min_read_length` for heavily
fragmented libraries, and raise `damage_max_read_length` only if every lane of
every sample was sequenced longer than it.

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
| `hit_max_dup_shallow` | `6.0` | shallow (`λ` < split): `dup` < threshold |
| `hit_min_cov_deep` | `0.02` | deep (`λ` ≥ split): `cov` > threshold |
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
| `{sample_id}.damage_profile_stratified.tsv` | Per-species absolute-position profiles split by read-length stratum |
| `{sample_id}.damage_model.tsv` | Per-species per-base damage rates from k-mer window deconvolution |
| `{sample_id}.coverage.tsv` | Per-taxon coverage and evenness statistics |
| `{sample_id}.summary.tsv.gz` | This sample's slice of the integrated summary, hit flags included |
| `{sample_id}.damage_profile.pdf` | Absolute-position damage plots (hit species) |
| `{sample_id}.damage_summary.pdf` | Damage biplot across all profiled taxa |

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

## Model-based damage estimation

`damage_score` is the drop from the terminal k-mer to a plateau further into the
read. That plateau is not always reached. A k-mer spans `k` bases, so for a read
of length `L` the k-mer at index `j` covers bases `j … j+k-1`, and no k-mer
clears both termini unless `L ≥ k + 2·(decay length)`. For fragments not much
longer than `k` the profile is a U whose two arms are the two read ends, its
minimum at `j = (L-k)/2`. The "plateau" is then still damaged, so `damage_score`
is a lower bound whose magnitude depends on read length — which makes any pooled
value depend on the sample's fragment length distribution.

`{sample_id}.damage_model.tsv` sidesteps this by treating the k-mer window as a
known convolution and inverting it. Per-base mismatch probability:

```
d_i = interior_rate
    + damage_rate_5prime · exp(-i / decay_5prime)
    + damage_rate_3prime · exp(-(L-1-i) / decay_3prime)
```

KrakenUniq matches k-mers exactly, so a k-mer is classified only if every base
in it matches, and `P(k-mer at 5' index j unclassified) = 1 - Π_{i=j}^{j+k-1} (1 - d_i)`.
The 3′-indexed counts are the same `d` vector read from the other end, so both
ends and every read-length stratum are functions of one five-parameter fit. No
plateau is needed (`interior_rate` is estimated), no read-length weighting is
needed (the parameters describe the molecules, not which lengths were
sequenced), short reads become informative rather than a nuisance, and the
output is a per-base rate comparable to mapDamage.

The read-length composition inside each stratum is recovered from the
per-position counts themselves — a read reaches index `j` only if it has more
than `j` k-mers, so neighbouring positions differ by the number of reads with
exactly that many. Reads with `n_kmers ≥ damage_max_pos` are censored into the
last position and represented by a single length; setting `damage_max_pos` at or
above `damage_max_read_length - k + 1` removes that approximation.

This is emitted **alongside** `damage_score`, never in place of it. Hit criteria
and plotting still use `damage_score`.

| Column | Meaning |
|---|---|
| `interior_rate` | Per-base mismatch floor (sequencing error + divergence) |
| `damage_rate_5prime` / `_3prime` | Terminal damage amplitude above the floor |
| `decay_5prime` / `_3prime` | Exponential decay length, bp |
| `terminal_rate_5prime` / `_3prime` | `interior_rate + damage_rate`, the rate at the terminal base |
| `damage_model_pvalue_5prime` / `_3prime` | One-sided test that the amplitude exceeds zero — **not usable for detection, see below** |
| `chi2_df` | Fit quality; values in the hundreds mean the exponential shape is a poor fit at that depth |
| `converged`, `n_obs`, `n_strata_used` | Fit provenance |

**Do not use `damage_model_pvalue_*` for detection.** k-mers within a read
overlap, so the binomial likelihood understates variance; standard errors are
scaled by `sqrt(chi2/df)` as a quasi-likelihood correction. That correction also
absorbs shape misspecification, and for taxa with many reads the single
exponential is a visible approximation, so `chi2/df` reaches the hundreds and
the standard errors inflate with it. The result is a test that gets *less*
sensitive as evidence accumulates. Measured on DA195 (species rank, fraction
reaching p < 0.05, against a 5% null expectation):

| reads | `damage_pvalue` (plateau) | `damage_model_pvalue_5prime` |
|---|---|---|
| 100–200 | 6.4% | 4.7% |
| 200–400 | 6.3% | 0.7% |
| 400–1000 | 5.8% | 0.8% |
| 1000–5000 | 14.8% | 4.1% |
| 5000+ | 60.7% | **0.0%** |

The plateau test separates from the null above ~1000 reads; the model test never
does, and collapses to zero exactly where the signal is strongest. Use
`damage_pvalue` for significance and hit criteria, which is what the workflow
does.

**The rate estimates are the deliverable.** `damage_rate_5prime` / `_3prime` and
`interior_rate` are stable where `damage_score` is not: on synthetic controls
they move ≈10% across stratum and `damage_max_pos` choices that move
`damage_score` by ~280%, and on the undamaged control the amplitudes go to zero.
Read `chi2_df` as a fit-quality flag, and note that `decay_5prime` / `_3prime`
frequently rail at their bounds on real data — the decay length is not well
identified by a single exponential, so treat the amplitudes, not the decays, as
the interpretable parameters.

**Detection limits at low depth.** `damage_score` is a difference of two
binomial proportions, so the smallest detectable value scales as 1/sqrt(reads):
roughly 6.8 percentage points at 100 reads, 4.8 at 200, 3.0 at 500, 2.1 at 1000
(for a ~15% plateau). Below ~1000 reads the damage test is at its null rate, and
taxa under `damage_min_reads` never enter the summary at all, since a
`damage_pvalue` is required. Low-read candidates therefore have to be triaged on
`kmers`, `cov` and abundance rather than damage significance — but note that at
~100 reads `kmers` is only 2–4x the read count, so it measures how much evidence
there is, not how evenly it is spread.

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
