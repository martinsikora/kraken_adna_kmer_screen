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
                                            ──► summary.pdf

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

### Per-species genome lengths (optional)

Needed only for the `evenness_depth` column in `coverage.tsv`. Also run once per
database, from its `library_seq_info.tsv`:

```bash
python scripts/build_species_genome_lengths.py \
    --library-seq-info /path/to/krakendb/library_seq_info.tsv \
    --out /path/to/krakendb/kraken_adna_kmer_screen_db/species.genome_lengths.tsv
```

Genome length is the median across assemblies of the summed sequence lengths
within an assembly. Summing every sequence for a species would multiply the
genome by the number of assemblies; averaging sequence lengths would divide it
by the number of replicons. Set `species_genome_lengths` in the dataset config
to the output path, or leave it empty to omit the column.

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
| `shortlist_guarantee` | `true` | Exempt well-covered taxa from the fast-mode shortlist |
| `shortlist_guarantee_min_reads` | `70` | Reads a species needs to qualify for the exemption |

#### Why the shortlist needs a guarantee

In `fit_mode: fast` the candidate shortlist keeps only the `max_candidates`
genera with the largest **absolute** shared k-mer mass. That is a magnitude,
not a measure of evidence, and a genus row is the unnormalised sum of its
species' reference weights, so a small viral genome is penalised twice: once
for its size and once for having few species summed into its row. Anything cut
never reaches NNLS, gets no `abundance.tsv` row, and so cannot appear in the
hit table -- while its `damage_stats.tsv` and `coverage.tsv` rows survive
intact and look perfectly healthy. The loss is silent.

Measured on a 17-sample screen: *Human mastadenovirus C* in one sample carried
138 reads at 18.4% 5' damage (p = 3e-5), a fitted damage rate of 0.22 and a
duplication of 3.2, yet its genus ranked 1439 of 3095 candidates and was
discarded. Across all samples 128 viral species had damage statistics but no
abundance row.

With `shortlist_guarantee` enabled, any species in `coverage.tsv` with at least
`shortlist_guarantee_min_reads` reads that also passes the depth-aware evenness
test -- the same `hit_evenness_lambda_split` / `hit_max_dup_shallow` /
`hit_min_cov_deep` thresholds used for hit selection, so the two cannot drift
apart -- is **added** to the shortlist along with its genus, rather than made
to compete for a place in it. Nothing the mass ranking kept is displaced.

The evenness test is what makes this affordable. A read-count floor alone
admits most of a diverse sample (+1934 genera in one case); adding the evenness
requirement holds it to a median +225. Measured cost on two samples, base
versus guarantee, back to back on the same host: +598 candidates cost +16%
runtime, +125 cost +12%. Existing results are untouched -- no species dropped,
`within_genus_relative_abundance` identical to the last bit, and shifts of
under 1e-3 in `genus_relative_abundance` from the wider NNLS.

Normalising the shortlist score was tried first and rejected. Ranking by
fraction-of-reference-matched (L1) or cosine (L2) pushed the same viral genera
*further* down -- a multi-species viral genus sums every species into its row
while the sample matches one, so its matched fraction is low -- and evicted 101
of 434 existing hits. Raising `max_candidates` to remove the truncation
entirely is the other option; it is correct but expensive, and an unbounded fit
on these samples had not finished after an hour against ~4 minutes shortlisted.

Set `shortlist_guarantee: false` to restore the previous behaviour exactly.

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
| `{sample_id}.coverage.tsv` | Per-taxon coverage and evenness statistics, incl. `evenness_depth` |
| `{sample_id}.summary.tsv.gz` | This sample's slice of the integrated summary, hit flags included |
| `{sample_id}.damage_profile.pdf` | Absolute-position damage plots (hit species) |
| `{sample_id}.summary.pdf` | Damage biplot across all profiled taxa |

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
| `interior_rate` | Per-base mismatch floor from the damage model (see below) |
| `damage_rate_5prime` / `_3prime` | Per-base terminal damage rate, above that floor |
| `hit_criteria_flag` | Semicolon-delimited passing criteria tokens from: `damage_rate`, `evenness_index`, `within_genus_relative_abundance`, `classified_rate` |

`damage_rate` combines two tests: `damage_pvalue < hit_max_damage_pvalue` **and**
`damage_rate_5prime <= hit_max_damage_rate` (default 0.4). Deamination in a
single-stranded overhang saturates near 0.5 per base, so a higher fitted rate is
not a damage profile but a taxon whose reads mismatch the reference throughout,
with the terminal excess an artefact of misassignment. A taxon with no model fit
has NaN and is judged on the p-value alone rather than rejected. On the
17-sample dev screen the cutoff vetoes 1322 rows that have a significant
p-value, 48 of which would otherwise pass the other criteria — concentrated in
the environmental samples (34 in Kolyma_River, 9 in Saqqaq), and many sitting at
the 0.9 parameter bound with damage scores of 0.5–0.98.

The three rate columns are joined from each sample's `damage_model.tsv` by a
left merge, so a taxon with no qualifying stratum keeps its row with NaN rates
(about 15% of rows at the default threshold). `damage_rate_5prime` feeds the
`damage_rate` criterion above; `interior_rate` and `damage_rate_3prime` are
annotations. The terminal rate is not carried, being exactly
`interior_rate + damage_rate_*`.

`damage_model_min_stratum_reads` is a **per-stratum** floor, so lowering it does
not rescue every low-read taxon: a taxon with 90 reads split 64/26 across two
strata still has no qualifying stratum. On the 17-sample dev screen, 42 of the
54 rate-less hits are missing for that reason and rest on the p-value alone.

They separate two artifact classes that `damage_score` alone conflates: an
impossible deamination rate (*Hydrogenimonas cancrithermarum* at
`damage_rate_5prime` 0.59, *Arcobacter venerupis* at 0.34 — both with near-zero
`interior_rate`), versus reads that do not match the reference away from the
termini (`interior_rate` 1.9–2.6%, the misassignment signature).

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

## Depth-normalised evenness (`evenness_depth`)

`evenness_index = cov / (1 - exp(-dup·cov))` reduces to `1/dup` at low depth —
measured across seven samples it is exactly that (median evenness × median dup =
1.000–1.009) — and the fraction of taxa passing `> 0.5` ranges from 0.02% to 42%
with no biological difference behind it.

That reduction is not a defect in the formula. With `N` k-mer observations over
`G` positions, expected breadth is `N/G`, observed is `unique/G`, and the ratio
is `unique/N = 1/dup`. The denominator cancels, so **any consistently normalised
variant returns the same number**. Renormalising with the database's own
per-species k-mer counts (`database.kdb.counts`, whose clade sums reproduce
`kmers/cov` exactly) reproduces `evenness_index` to seven decimal places —
Spearman 1.000000, max absolute difference 9.6e-07. Changing the denominator
achieves nothing.

`coverage.tsv` therefore carries a second score that deliberately breaks that
consistency, taking depth from bases sequenced rather than k-mer observations:

```
depth_estimate = reads · mean_read_length / genome_length
evenness_depth = cov / (1 - exp(-depth_estimate))
```

| criterion | cross-sample spread in pass rate |
|---|---|
| `evenness_index > 0.5` | 1884× |
| `evenness_depth > 0.5` | **8.9×** |

It also carries signal the original does not. Against `interior_rate` from the
damage model — the model's estimate of how badly reads match the reference away
from the termini, so a proxy for misassignment — Spearman correlations are:

| | `evenness_depth` | `evenness_index` | k-mer-count normalised | `cov` | `dup` |
|---|---|---|---|---|---|
| 018345 (n=5208) | **−0.197** | +0.042 | +0.042 | +0.047 | −0.042 |
| DA195 (n=2252) | **−0.160** | +0.007 | +0.007 | +0.069 | −0.010 |

`evenness_index`, `cov` and `dup` are all uncorrelated with misassignment, and
two of them carry the wrong sign. The k-mer-count-normalised column is the
consistency check above — identical to `evenness_index`, as the cancellation
requires.

**What `evenness_depth` actually measures.** Since every consistently
normalised variant collapses to `1/dup`, the signal comes from the one
inconsistent thing this score does: a bases-sequenced numerator against a
discriminative-k-mer denominator. That makes it closer to *unique discriminative
k-mers per base sequenced* — `1/dup` scaled by the fraction of each read's
k-mers that discriminate the taxon. That fraction is small for taxa with close
relatives in the database, which are exactly the misassignment-prone ones, so
the score blends coverage evenness with taxonomic distinctiveness. For screening
that blend behaves better than evenness alone, but read it as a screening
statistic rather than a pure evenness measure.

**Requirements.** Needs `species_genome_lengths` in the config (built once per
database by `scripts/build_species_genome_lengths.py` from the database's
`library_seq_info.tsv`) and per-unit summaries carrying `mean_read_length`.
Either missing leaves the column NaN rather than guessing — so samples screened
before `mean_read_length` was added need a re-screen to populate it.

**`kmer_set_ratio` — read this before using the score.** `cov` is breadth
against the union of taxon-discriminative k-mers across every strain in the
database, not against a genome, so `evenness_depth` pairs a `cov` numerator with
a genome denominator. `kmer_set_ratio = (kmers/cov) / genome_length` reports how
far apart those are; near 1 the score is sound, far from 1 it is not. About 83%
of taxa fall in 0.5–2. The failures are informative:

- **Hepatitis B virus** sits at ~400 (9950 strains in the database). In DA195 it
  is at 12× depth — the best-covered organism in the sample — but `cov`
  saturates at 0.004, so `evenness_depth` wrongly reports 0.004.
- **Yersinia pestis** sits at ~0.13, most of its genome being shared with
  *Y. pseudotuberculosis* and assigned above the species node.

Filter on `kmer_set_ratio` before comparing `evenness_depth` across species.

This cannot be fixed from the database as it stands. It needs the number of
discriminative k-mers in a *single* representative genome, and the per-taxon
counts do not carry it: k-mers shared across strains sit at the species node
while strain nodes hold only strain-specific k-mers, so the per-strain counts
are tiny and unrelated to genome size (Hepatitis B virus 64, *Y. pestis* 207).
Recovering it would mean re-deriving k-mer sets per genome from the database.

`evenness_depth` is an additional column and is **not** used by any hit
criterion; `evenness_index` remains the criterion.

---

## Damage visualisation

Three PDFs are produced per sample:

- **`damage_profile.pdf`** — Absolute-position profiles (0 = terminal k-mer) for the
  5′ and 3′ ends of each hit species. Shaded plateau region and per-end damage scores
  are annotated. One page per species.

- **`summary.pdf`** — Scatter plot of all profiled taxa. X axis: baseline
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
