#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# plot_damage_summary.R
#
# Summary biplot of aDNA damage across all taxa from workflow damage TSVs.
#
# One point per taxon:
#   X axis : baseline classified k-mer rate  (1 - plateau_frac_unc)
#   Y axis : damage rate (%)  (damage_score * 100; negative scores clamped to 0)
#   Size   : log10(n_reads)
#   Fill   : --color-by evenness (default) - evenness_index on mako reversed,
#            so dark = high evenness; or --color-by pvalue for -log10(p)
#   Shape  : filled down-triangle = damage-significant, circle = not
#   Outline: black = all four hit criteria passed, none otherwise
#   Labels : ggrepel with leader segments
#
# Colouring by evenness adds information the rest of the plot does not carry:
# both axes and the marker shape are damage-derived, whereas evenness separates
# genome-wide coverage from reads piled onto a conserved region. Evenness is
# read from the per-sample coverage.tsv, which covers every plotted taxon; the
# integrated summary only retains taxa with complete evidence of all four kinds
# and would leave a large fraction uncoloured.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(ggrepel)
  library(scales)
})

script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "."
}
source(file.path(script_dir(), "plot_common.R"))

ALL_HIT_FLAG_TOKENS <- c(
  "damage_pvalue", "evenness_index",
  "within_genus_relative_abundance", "classified_rate"
)

SPEC <- list(
  stats               = list(type = "character", nargs = 1,   default = NULL),
  output              = list(type = "character", nargs = 1,   default = NULL),
  coverage            = list(type = "character", nargs = 1,   default = NULL),
  hits                = list(type = "character", nargs = 1,   default = NULL),
  sample_id           = list(type = "character", nargs = 1,   default = NULL),
  hits_required_flags = list(type = "character", nargs = "+", default = character(0)),
  pvalue_threshold    = list(type = "double",    nargs = 1,   default = 0.05),
  min_reads           = list(type = "integer",   nargs = 1,   default = 10L),
  max_keys            = list(type = "integer",   nargs = 1,   default = 200L),
  max_x               = list(type = "double",    nargs = 1,   default = NULL),
  color_by            = list(type = "character", nargs = 1,   default = "evenness"),
  no_label_repel      = list(type = "flag",      nargs = 1,   default = FALSE)
)

args <- parse_cli(commandArgs(trailingOnly = TRUE), SPEC)
stopifnot(!is.null(args$stats), !is.null(args$output))

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

stats <- suppressWarnings(readr::read_tsv(args$stats, col_types = readr::cols(
  .default = readr::col_guess(),
  species_name = readr::col_character(),
  end = readr::col_character()
), progress = FALSE))

df <- stats %>%
  filter(.data$end == "5prime", .data$n_reads >= args$min_reads) %>%
  mutate(
    x           = 1 - .data$plateau_frac_unc,
    y           = pmax(.data$damage_score, 0) * 100,
    neg_log10_p = -log10(pmax(.data$damage_pvalue, 1e-300)),
    log10_n     = log10(pmax(.data$n_reads, 1)),
    significant = .data$damage_pvalue < args$pvalue_threshold & .data$damage_score > 0,
    label       = dplyr::coalesce(as.character(.data$species_name), "")
  )
if ("taxid" %in% names(df)) {
  df$label <- ifelse(nzchar(df$label), df$label, as.character(df$taxid))
}

if (nrow(df) == 0L) {
  write_empty_plot(
    args$output,
    "No damage-summary points available",
    "No taxa passed plotting filters (end=5prime and min-reads threshold)."
  )
  quit(save = "no", status = 0)
}

# evenness for colour: prefer per-sample coverage.tsv, fall back to the summary
load_evenness <- function(path, sample_id) {
  if (is.null(path) || is.null(sample_id)) return(NULL)
  d <- tryCatch(
    suppressWarnings(readr::read_tsv(path, col_types = readr::cols(
      .default = readr::col_guess(), sample_id = readr::col_character()
    ), progress = FALSE)),
    error = function(e) {
      message("WARNING: could not read evenness from ", path, ": ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(d) || !"evenness_index" %in% names(d)) return(NULL)
  if ("sample_id" %in% names(d)) {
    d <- d[as.character(d$sample_id) == as.character(sample_id), , drop = FALSE]
  }
  if ("rank" %in% names(d)) {
    d <- d[tolower(trimws(d$rank)) == "species", , drop = FALSE]
  }
  name_col <- if ("species_name" %in% names(d)) "species_name" else "tax_name"
  if (!name_col %in% names(d)) return(NULL)
  d <- d[!is.na(d[[name_col]]), , drop = FALSE]
  d <- d[!duplicated(d[[name_col]]), , drop = FALSE]
  setNames(as.numeric(d$evenness_index), d[[name_col]])
}

if (identical(args$color_by, "evenness")) {
  ev <- load_evenness(if (!is.null(args$coverage)) args$coverage else args$hits,
                      args$sample_id)
  df$evenness_index <- if (is.null(ev)) NA_real_ else unname(ev[df$label])
  n_missing <- sum(is.na(df$evenness_index))
  if (n_missing > 0L) {
    message("NOTE: ", n_missing, " of ", nrow(df),
            " taxa have no evenness_index (drawn grey)")
  }
}

# annotation set and full-hit set
required_flags <- resolve_required_hit_flags(args$hits_required_flags)
sel <- select_hit_species(args$hits, args$sample_id, required_flags, args$max_keys)
if (sel$n_truncated > 0L) {
  message("WARNING: selected hit species truncated to ", length(sel$species),
          " (dropped ", sel$n_truncated, " by --max-keys).")
}
full <- select_hit_species(args$hits, args$sample_id, ALL_HIT_FLAG_TOKENS, 0L)

df$is_full <- df$label %in% full$species
to_label <- df[df$label %in% sel$species, , drop = FALSE]

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

if (identical(args$color_by, "evenness")) {
  df$fill_value <- df$evenness_index
  fill_scale <- scale_fill_viridis_c(
    option = "mako", direction = -1, limits = c(0, 1),
    na.value = "#cccccc", name = "evenness_index"
  )
} else {
  df$fill_value <- df$neg_log10_p
  fill_scale <- scale_fill_viridis_c(
    option = "mako", direction = -1,
    na.value = "#cccccc", name = expression(-log[10](p))
  )
}

# shapes 21/25 take a separate fill and outline colour, so the evenness scale
# and the full-hit outline can be encoded independently
df$shape <- ifelse(df$significant, 25L, 21L)

size_breaks <- c(100, 1e3, 1e4, 1e5)

p <- ggplot(df, aes(x = .data$x, y = .data$y)) +
  geom_hline(yintercept = 0, linetype = "dashed", colour = "grey60",
             linewidth = 0.3) +
  geom_point(
    aes(fill = .data$fill_value, size = .data$log10_n,
        shape = .data$shape, colour = .data$is_full),
    alpha = 0.8, stroke = 0.35
  ) +
  fill_scale +
  scale_shape_identity() +
  # "transparent", not NA: a row whose colour maps to NA is dropped by ggplot
  # entirely, which would silently plot only the outlined points.
  scale_colour_manual(
    values = c(`TRUE` = "black", `FALSE` = "transparent"),
    na.value = "transparent", guide = "none"
  ) +
  scale_size_continuous(
    name = "reads",
    range = c(0.6, 7),
    breaks = log10(size_breaks),
    labels = scales::comma(size_breaks)
  ) +
  labs(
    x = "Baseline classified k-mer rate  (1 − plateau unclassified)",
    y = "Damage rate (%)",
    title = "aDNA damage summary"
  ) +
  theme_screen()

if (!is.null(args$max_x)) p <- p + coord_cartesian(xlim = c(NA, args$max_x))

if (nrow(to_label) > 0L) {
  p <- if (isTRUE(args$no_label_repel)) {
    p + geom_text(data = to_label, aes(label = .data$label),
                  size = 2.3, colour = "#333333", hjust = 0, vjust = 0,
                  nudge_x = 0.004, nudge_y = 0.4)
  } else {
    p + geom_text_repel(
      data = to_label, aes(label = .data$label),
      size = 2.3, colour = "#333333",
      segment.colour = "grey55", segment.size = 0.25,
      min.segment.length = 0, box.padding = 0.35, point.padding = 0.2,
      max.overlaps = Inf, seed = 42
    )
  }
}

ggsave(args$output, p, width = 9, height = 6.5, device = pdf_device())
message("Saved ", args$output)
