#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# plot_damage_fractional.R
#
# Fractional-position aDNA damage profiles from the pre-computed profiles
# written by aggregate_sample.py (*_damage_fractional_profile.tsv).
#
#   X axis : fractional position along the read (0 = 5', 1 = 3')
#   Y axis : fraction of unclassified k-mers (%)
#
# One page per taxon, strata stacked in one column with free y scales.
# Stratifying by read length separates genuine terminal damage (present in all
# strata) from the read-length artifact, where short reads have overlapping
# damage zones and no flat interior.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
})

script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "."
}
source(file.path(script_dir(), "plot_common.R"))

DEFAULT_STRATA <- c("31-40", "41-55", "56-75", "76-100")
STRATUM_COLOURS <- c("#d73027", "#fc8d59", "#4575b4", "#313695")
SAMPLE_COLOURS <- c("#d6604d", "#2166ac", "#4dac26", "#8073ac")
SAMPLE_LINETYPES <- c("solid", "dashed", "dotdash", "dotted")

SPEC <- list(
  fractional          = list(type = "character", nargs = "+", default = character(0)),
  output              = list(type = "character", nargs = 1,   default = NULL),
  label               = list(type = "character", nargs = "+", default = character(0)),
  strata              = list(type = "character", nargs = "+", default = character(0)),
  taxid               = list(type = "integer",   nargs = 1,   default = NULL),
  species             = list(type = "character", nargs = 1,   default = NULL),
  global              = list(type = "character", nargs = 1,   default = NULL),
  hits                = list(type = "character", nargs = 1,   default = NULL),
  sample_id           = list(type = "character", nargs = 1,   default = NULL),
  hits_required_flags = list(type = "character", nargs = "+", default = character(0)),
  max_keys            = list(type = "integer",   nargs = 1,   default = 200L),
  pvalue_threshold    = list(type = "double",    nargs = 1,   default = 0.05)
)

args <- parse_cli(commandArgs(trailingOnly = TRUE), SPEC)
stopifnot(length(args$fractional) > 0, !is.null(args$output))

strata <- if (length(args$strata) > 0) args$strata else DEFAULT_STRATA
n_samp <- length(args$fractional)
labels <- if (length(args$label) == n_samp) args$label else paste("Sample", seq_len(n_samp))
colours <- SAMPLE_COLOURS[(seq_len(n_samp) - 1) %% length(SAMPLE_COLOURS) + 1]
ltypes  <- SAMPLE_LINETYPES[(seq_len(n_samp) - 1) %% length(SAMPLE_LINETYPES) + 1]

# ---------------------------------------------------------------------------
# Which taxa to plot
# ---------------------------------------------------------------------------

resolve_keys_from_global <- function(path, pthr) {
  if (is.null(path)) return(character(0))
  df <- suppressWarnings(readr::read_tsv(path, col_types = readr::cols(
    .default = readr::col_guess(), species_name = readr::col_character()
  ), progress = FALSE))
  if (nrow(df) == 0L) return(character(0))
  key_col <- if ("species_name" %in% names(df)) "species_name" else "taxid"
  if ("damage_pvalue" %in% names(df)) {
    sig <- df$damage_score > 0 & df$damage_pvalue < pthr
    sig_keys <- unique(stats::na.omit(df[[key_col]][sig]))
    if (length(sig_keys) > 0) {
      cand <- df[df[[key_col]] %in% sig_keys, , drop = FALSE]
      agg <- aggregate(cand$damage_pvalue, by = list(key = cand[[key_col]]),
                       FUN = function(v) mean(v, na.rm = TRUE))
      agg <- agg[order(agg$x), , drop = FALSE]
      message("Auto-selected ", nrow(agg), " taxa with damage_pvalue < ", pthr, ".")
      return(agg$key)
    }
    message("No taxa pass p < ", pthr, ". Falling back to top 3 by score.")
  }
  agg <- aggregate(df$damage_score, by = list(key = df[[key_col]]),
                   FUN = function(v) mean(v, na.rm = TRUE))
  utils::head(agg[order(-agg$x), , drop = FALSE]$key, 3)
}

use_hits <- !is.null(args$hits) && !is.null(args$sample_id)
if (use_hits) {
  sel <- select_hit_species(args$hits, args$sample_id,
                            resolve_required_hit_flags(args$hits_required_flags),
                            args$max_keys)
  keys <- sel$species
  if (length(keys) == 0L) {
    message("WARNING: no hit species matched all required hit flags for this sample.")
  }
  if (sel$n_truncated > 0L) {
    message("WARNING: selected hit species truncated to ", length(keys),
            " (dropped ", sel$n_truncated, " by --max-keys).")
  }
} else if (!is.null(args$taxid)) {
  keys <- args$taxid
} else if (!is.null(args$species)) {
  keys <- args$species
} else {
  keys <- resolve_keys_from_global(args$global, args$pvalue_threshold)
  if (args$max_keys > 0 && length(keys) > args$max_keys) {
    message("WARNING: auto-selected taxa truncated to ", args$max_keys,
            " (dropped ", length(keys) - args$max_keys, " by --max-keys).")
    keys <- keys[seq_len(args$max_keys)]
  }
}

if (length(keys) == 0L) {
  write_empty_plot(
    args$output,
    "No fractional damage plots were generated",
    "No taxa were selected from --hits, explicit args, or --global fallback."
  )
  quit(save = "no", status = 0)
}

# ---------------------------------------------------------------------------
# Load fractional tables
# ---------------------------------------------------------------------------

wanted <- c("taxid", "species_name", "stratum", "bin", "n_reads",
            "n_total", "n_unclassified")

load_fractional <- function(path) {
  d <- suppressWarnings(readr::read_tsv(path, col_types = readr::cols(
    .default = readr::col_guess(),
    species_name = readr::col_character(),
    stratum = readr::col_character()
  ), progress = FALSE))
  d[, intersect(wanted, names(d)), drop = FALSE]
}

tables <- lapply(args$fractional, load_fractional)

key_col_for <- function(d, key) if (is.character(key)) "species_name" else "taxid"

available <- unique(unlist(lapply(tables, function(d) {
  if (is.character(keys[[1]]) && "species_name" %in% names(d)) {
    as.character(stats::na.omit(d$species_name))
  } else if ("taxid" %in% names(d)) {
    stats::na.omit(d$taxid)
  } else NULL
})))
if (length(available) > 0) {
  kept <- keys[keys %in% available]
  if (length(kept) < length(keys)) {
    message("WARNING: dropped ", length(keys) - length(kept),
            " selected taxa with no fractional rows.")
  }
  keys <- kept
}

if (length(keys) == 0L) {
  write_empty_plot(
    args$output,
    "No fractional damage plots were generated",
    "No selected taxa had matching rows in fractional profile tables."
  )
  quit(save = "no", status = 0)
}

# ---------------------------------------------------------------------------
# Per-page plot
# ---------------------------------------------------------------------------

build_page <- function(key) {
  rows <- list()
  for (i in seq_len(n_samp)) {
    d <- tables[[i]]
    kc <- key_col_for(d, key)
    if (!kc %in% names(d)) next
    sub <- d[d[[kc]] == key & d$stratum %in% strata, , drop = FALSE]
    if (nrow(sub) == 0L) next
    n_bins <- max(sub$bin, na.rm = TRUE) + 1L
    sub <- sub[sub$n_total > 0, , drop = FALSE]
    if (nrow(sub) == 0L) next
    reads_by_stratum <- tapply(sub$n_reads, sub$stratum, function(v) v[1])
    rows[[length(rows) + 1L]] <- data.frame(
      stratum = sub$stratum,
      frac = (sub$bin + 0.5) / n_bins,
      pct  = sub$n_unclassified / sub$n_total * 100,
      series = sprintf("%s  (n=%s)", labels[i],
                       format(as.integer(reads_by_stratum[sub$stratum]), big.mark = ",")),
      sample_i = i,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) return(NULL)
  dat <- dplyr::bind_rows(rows)
  dat$stratum <- factor(dat$stratum, levels = strata,
                        labels = paste0(sub("-", "–", strata), " bp"))
  dat <- dat[!is.na(dat$stratum), , drop = FALSE]
  if (nrow(dat) == 0L) return(NULL)
  dat <- dat[order(dat$stratum, dat$series, dat$frac), , drop = FALSE]

  series_levels <- unique(dat$series)
  pal <- colours[vapply(series_levels, function(s) dat$sample_i[match(s, dat$series)], integer(1))]
  lty <- ltypes[vapply(series_levels, function(s) dat$sample_i[match(s, dat$series)], integer(1))]
  names(pal) <- series_levels; names(lty) <- series_levels
  dat$series <- factor(dat$series, levels = series_levels)

  ggplot(dat, aes(x = .data$frac, y = .data$pct,
                  colour = .data$series, linetype = .data$series)) +
    geom_vline(xintercept = 0.5, linetype = "dotted", colour = "grey60",
               linewidth = 0.3) +
    geom_line(linewidth = 0.7, alpha = 0.9) +
    scale_colour_manual(values = pal, name = NULL) +
    scale_linetype_manual(values = lty, name = NULL) +
    # Stacked in one column with per-stratum y ranges, not anchored at zero.
    # Strata are independent measurements at different read lengths, so free
    # scales cost little; anchoring at zero left the curve using a fifth to a
    # third of each panel whenever the baseline unclassified rate was high.
    facet_grid(stratum ~ ., scales = "free_y", switch = "y") +
    scale_x_continuous(limits = c(0, 1), expand = expansion(mult = 0.01)) +
    scale_y_continuous(expand = expansion(mult = 0.08)) +
    labs(
      title = "aDNA damage profiles — fractional position by read length",
      subtitle = as.character(key),
      x = "Fractional position (0 = 5′, 1 = 3′)",
      y = "Unclassified k-mers (%)",
      caption = paste(
        "Short reads (31-40 bp): damage zones overlap across the entire read,",
        "no flat interior. Longer reads show a flat central plateau."
      )
    ) +
    theme_screen() +
    theme(legend.position = "top", legend.direction = "vertical",
          panel.spacing.y = unit(0.5, "lines"), strip.placement = "outside")
}

# ---------------------------------------------------------------------------
# Write multi-page PDF
# ---------------------------------------------------------------------------

open_pdf_device(args$output, width = 7.5,
                height = max(5, 1.9 * length(strata) + 1.6))

pages <- 0L
for (k in keys) {
  p <- build_page(k)
  if (is.null(p)) next
  print(p)
  pages <- pages + 1L
}
if (pages == 0L) {
  print(
    ggplot() +
      annotate("text", x = 0.5, y = 0.6,
               label = "No fractional damage plots were generated",
               size = 4.6, fontface = "bold") +
      annotate("text", x = 0.5, y = 0.42,
               label = "Selected taxa had no matching rows in the fractional profile table.",
               size = 3.4) +
      xlim(0, 1) + ylim(0, 1) + theme_void()
  )
  pages <- 1L
}
invisible(dev.off())
message("Saved ", args$output, " (", pages, " page(s))")
