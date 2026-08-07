#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# plot_damage_fractional.R
#
# Fractional-position aDNA damage profiles from the pre-computed profiles
# written by aggregate_sample.py (*_damage_fractional_profile.tsv).
#
#   X axis : fractional position along the read (0 = 5', 1 = 3')
#   Colour : 5' half red, 3' half blue (mapDamage convention)
#   Y axis : fraction of unclassified k-mers (%)
#
# One page per taxon, strata stacked in one column with free y scales.
# Stratifying by read length separates genuine terminal damage (present in all
# strata) from the read-length artifact, where short reads have overlapping
# damage zones and no flat interior.
#
# --strata may request wider bins than the profile table stores: a requested
# stratum spanning several stored ones is summed on read (see map_strata), so
# regrouping read lengths needs no re-screening.
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

# matches dataset_example/config/config.yaml; the workflow always passes --strata
DEFAULT_STRATA <- c("31-55", "56-75", "76-100")
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

#' Map stored strata onto requested strata.
#'
#' A requested stratum may be the union of several stored ones: the profile
#' table holds whatever strata screen_unit accumulated, and asking for a wider
#' bin (e.g. "31-55" over stored "31-40" and "41-55") is exactly a sum. Every
#' read falls in exactly one stored stratum and fractional binning depends only
#' on that read's own k-mer positions, so n_total, n_unclassified and n_reads
#' are additive across stored strata with no re-screening required.
#'
#' A stored stratum that only partially overlaps a requested one cannot be
#' split and is dropped with a warning.
map_strata <- function(stored, requested) {
  rng <- function(s) {
    parts <- strsplit(s, "-", fixed = TRUE)
    do.call(rbind, lapply(parts, function(p) as.numeric(p[1:2])))
  }
  sr <- rng(stored); rr <- rng(requested)
  out <- list()
  for (j in seq_along(requested)) {
    inside <- which(sr[, 1] >= rr[j, 1] & sr[, 2] <= rr[j, 2])
    partial <- which(sr[, 1] <= rr[j, 2] & sr[, 2] >= rr[j, 1])
    partial <- setdiff(partial, inside)
    if (length(partial) > 0L) {
      message("WARNING: stored stratum/strata ", paste(stored[partial], collapse = ", "),
              " partially overlap requested ", requested[j], "; cannot split, dropped.")
    }
    if (length(inside) == 0L) next
    if (length(inside) > 1L) {
      message("NOTE: requested stratum ", requested[j], " summed from stored ",
              paste(stored[inside], collapse = " + "), ".")
    }
    out[[length(out) + 1L]] <- data.frame(
      stored = stored[inside], requested = requested[j], stringsAsFactors = FALSE
    )
  }
  if (length(out) == 0L) return(data.frame(stored = character(0), requested = character(0)))
  dplyr::bind_rows(out)
}

build_page <- function(key) {
  rows <- list()
  reads <- list()
  for (i in seq_len(n_samp)) {
    d <- tables[[i]]
    kc <- key_col_for(d, key)
    if (!kc %in% names(d)) next
    sub <- d[d[[kc]] == key, , drop = FALSE]
    if (nrow(sub) == 0L) next

    m <- map_strata(unique(sub$stratum), strata)
    if (nrow(m) == 0L) next
    sub$requested <- m$requested[match(sub$stratum, m$stored)]
    sub <- sub[!is.na(sub$requested), , drop = FALSE]
    if (nrow(sub) == 0L) next

    n_bins <- max(sub$bin, na.rm = TRUE) + 1L

    # reads per requested stratum: n_reads is constant within a stored stratum,
    # so take it once per stored stratum and sum across the contributors
    per_stored <- sub[!duplicated(sub$stratum), c("stratum", "requested", "n_reads")]
    reads[[length(reads) + 1L]] <- stats::aggregate(
      per_stored$n_reads, by = list(requested = per_stored$requested), FUN = sum
    )

    agg <- stats::aggregate(
      cbind(n_total, n_unclassified) ~ requested + bin, data = sub, FUN = sum
    )
    agg <- agg[agg$n_total > 0, , drop = FALSE]
    if (nrow(agg) == 0L) next

    rows[[length(rows) + 1L]] <- data.frame(
      stratum  = agg$requested,
      frac     = (agg$bin + 0.5) / n_bins,
      pct      = agg$n_unclassified / agg$n_total * 100,
      sample_i = i,
      stringsAsFactors = FALSE
    )
  }
  if (length(rows) == 0L) return(NULL)
  dat <- dplyr::bind_rows(rows)
  reads <- dplyr::bind_rows(reads)

  # Read counts belong in the facet strip, not the series label: with one
  # sample and four strata a per-stratum series label produced four legend
  # entries for what is a single series.
  strip <- vapply(strata, function(s) {
    n <- sum(reads$x[reads$requested == s], na.rm = TRUE)
    if (n > 0) sprintf("%s bp  (n=%s)", sub("-", "–", s), format(n, big.mark = ","))
    else paste0(sub("-", "–", s), " bp")
  }, character(1))

  dat$stratum <- factor(dat$stratum, levels = strata, labels = strip)
  dat <- dat[!is.na(dat$stratum), , drop = FALSE]
  if (nrow(dat) == 0L) return(NULL)
  dat$series <- labels[dat$sample_i]
  dat <- dat[order(dat$stratum, dat$series, dat$frac), , drop = FALSE]

  series_levels <- unique(dat$series)
  lty <- ltypes[vapply(series_levels, function(s) dat$sample_i[match(s, dat$series)], integer(1))]
  names(lty) <- series_levels
  dat$series <- factor(dat$series, levels = series_levels)

  # Split each line at the midpoint so the 5' half is red and the 3' half blue,
  # matching plot_damage_profile.R and the mapDamage convention. Bin centres
  # never land exactly on 0.5, so a point is interpolated there and given to
  # both halves — otherwise the two segments would be drawn with a visible gap.
  dat <- do.call(rbind, lapply(split(dat, list(dat$stratum, dat$series), drop = TRUE),
    function(d) {
      d <- d[order(d$frac), , drop = FALSE]
      lo <- d[d$frac <= 0.5, , drop = FALSE]
      hi <- d[d$frac >  0.5, , drop = FALSE]
      if (nrow(lo) > 0L && nrow(hi) > 0L) {
        x1 <- lo$frac[nrow(lo)]; y1 <- lo$pct[nrow(lo)]
        x2 <- hi$frac[1];        y2 <- hi$pct[1]
        ym <- y1 + (y2 - y1) * (0.5 - x1) / (x2 - x1)
        mid_lo <- lo[nrow(lo), , drop = FALSE]; mid_lo$frac <- 0.5; mid_lo$pct <- ym
        mid_hi <- hi[1, , drop = FALSE];        mid_hi$frac <- 0.5; mid_hi$pct <- ym
        lo <- rbind(lo, mid_lo); hi <- rbind(mid_hi, hi)
      }
      if (nrow(lo)) lo$half <- "5prime"
      if (nrow(hi)) hi$half <- "3prime"
      rbind(lo, hi)
    }))
  dat$half <- factor(dat$half, levels = c("5prime", "3prime"))
  end_pal <- c(`5prime` = "#d6604d", `3prime` = "#2166ac")

  ggplot(dat, aes(x = .data$frac, y = .data$pct,
                  colour = .data$half, linetype = .data$series,
                  group = interaction(.data$series, .data$half))) +
    geom_vline(xintercept = 0.5, linetype = "dotted", colour = "grey60",
               linewidth = 0.3) +
    geom_line(linewidth = 0.7, alpha = 0.9) +
    # no colour guide: the x axis and midpoint rule already say which end is which
    scale_colour_manual(values = end_pal, guide = "none") +
    scale_linetype_manual(values = lty, name = NULL,
                          guide = if (n_samp > 1L) "legend" else "none") +
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
      # name the actual shortest stratum: it is configurable and may be a
      # union of stored strata, so a hardcoded "31-40 bp" goes stale
      caption = paste0(
        "Short reads (", sub("-", "\u2013", strata[1]), " bp): damage zones overlap ",
        "across the entire read, no flat interior. ",
        "Longer reads show a flat central plateau."
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
