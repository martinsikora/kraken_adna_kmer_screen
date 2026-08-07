#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# plot_damage_profile.R
#
# Absolute-position aDNA damage profiles from workflow damage TSV outputs.
#
#   X axis : position from read end (0 = terminal k-mer)
#   Y axis : fraction of unclassified k-mers (%)
#
# One page per taxon, faceted into 5' and 3' panels. Multiple samples can be
# overlaid by passing --profile/--global/--stats more than once.
#
# --stats is optional but recommended: it supplies the adaptive plateau window
# and per-end p-values computed by aggregate_sample.py.
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

SAMPLE_COLOURS <- c("#d6604d", "#2166ac", "#4dac26", "#8073ac")
SAMPLE_LINETYPES <- c("solid", "dashed", "dotdash", "dotted")

SPEC <- list(
  profile             = list(type = "character", nargs = "+", default = character(0)),
  global              = list(type = "character", nargs = "+", default = character(0)),
  stats               = list(type = "character", nargs = "+", default = character(0)),
  label               = list(type = "character", nargs = "+", default = character(0)),
  output              = list(type = "character", nargs = 1,   default = NULL),
  taxids              = list(type = "integer",   nargs = "+", default = integer(0)),
  species             = list(type = "character", nargs = "+", default = character(0)),
  hits                = list(type = "character", nargs = 1,   default = NULL),
  sample_id           = list(type = "character", nargs = 1,   default = NULL),
  hits_required_flags = list(type = "character", nargs = "+", default = character(0)),
  max_keys            = list(type = "integer",   nargs = 1,   default = 200L),
  max_pos             = list(type = "integer",   nargs = 1,   default = 20L),
  plateau_start       = list(type = "integer",   nargs = 1,   default = 3L),
  plateau_end         = list(type = "integer",   nargs = 1,   default = 10L),
  pvalue_threshold    = list(type = "double",    nargs = 1,   default = 0.05)
)

args <- parse_cli(commandArgs(trailingOnly = TRUE), SPEC)
stopifnot(length(args$profile) > 0, !is.null(args$output))

if (length(args$profile) != length(args$global)) {
  stop("--profile and --global must match in count", call. = FALSE)
}
if (length(args$stats) > 0 && length(args$stats) != length(args$profile)) {
  stop("--stats count must match --profile count", call. = FALSE)
}

n_samp <- length(args$profile)
labels <- if (length(args$label) == n_samp) args$label else paste("Sample", seq_len(n_samp))
colours <- SAMPLE_COLOURS[(seq_len(n_samp) - 1) %% length(SAMPLE_COLOURS) + 1]
ltypes  <- SAMPLE_LINETYPES[(seq_len(n_samp) - 1) %% length(SAMPLE_LINETYPES) + 1]

read_any <- function(p) suppressWarnings(readr::read_tsv(
  p, col_types = readr::cols(.default = readr::col_guess(),
                             species_name = readr::col_character()),
  progress = FALSE
))

profiles <- lapply(args$profile, read_any)
gdfs     <- lapply(args$global, read_any)
sdfs     <- if (length(args$stats) > 0) lapply(args$stats, read_any) else vector("list", n_samp)

# ---------------------------------------------------------------------------
# Which taxa to plot
# ---------------------------------------------------------------------------

resolve_keys <- function(gdfs, taxids, species, pthr) {
  if (length(species) > 0) return(species)
  if (length(taxids) > 0) return(taxids)

  combined <- dplyr::bind_rows(gdfs)
  has_taxid <- "taxid" %in% names(combined) && any(!is.na(combined$taxid))
  key_col <- if (has_taxid) "taxid" else "species_name"

  if ("damage_pvalue" %in% names(combined)) {
    sig <- combined$damage_score > 0 & combined$damage_pvalue < pthr
    sig_keys <- unique(stats::na.omit(combined[[key_col]][sig]))
    if (length(sig_keys) > 0) {
      cand <- combined[combined[[key_col]] %in% sig_keys, , drop = FALSE]
      agg <- aggregate(cand$damage_pvalue, by = list(key = cand[[key_col]]),
                       FUN = function(v) mean(v, na.rm = TRUE))
      agg <- agg[order(agg$x), , drop = FALSE]
      message("Auto-selected ", nrow(agg), " taxa with damage_pvalue < ", pthr, ".")
      return(agg$key)
    }
    message("No taxa pass p < ", pthr, " with positive damage score. ",
            "Falling back to top 3 by score.")
  }
  agg <- aggregate(combined$damage_score, by = list(key = combined[[key_col]]),
                   FUN = function(v) mean(v, na.rm = TRUE))
  agg <- agg[order(-agg$x), , drop = FALSE]
  utils::head(agg$key, 3)
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
} else {
  keys <- resolve_keys(gdfs, args$taxids, args$species, args$pvalue_threshold)
  if (length(args$taxids) == 0 && length(args$species) == 0 &&
      args$max_keys > 0 && length(keys) > args$max_keys) {
    message("WARNING: auto-selected taxa truncated to ", args$max_keys,
            " (dropped ", length(keys) - args$max_keys, " by --max-keys).")
    keys <- keys[seq_len(args$max_keys)]
  }
}

# drop keys with no profile rows anywhere
if (length(keys) > 0 && is.character(keys)) {
  available <- unique(unlist(lapply(profiles, function(p)
    if ("species_name" %in% names(p)) stats::na.omit(as.character(p$species_name)) else character(0))))
  if (length(available) > 0) {
    kept <- keys[keys %in% available]
    if (length(kept) < length(keys)) {
      message("WARNING: dropped ", length(keys) - length(kept),
              " selected species with no profile rows.")
    }
    keys <- kept
  }
}

if (length(keys) == 0L) {
  write_empty_plot(
    args$output,
    "No damage-profile plots were generated",
    "No taxa were selected from --hits or auto-selection criteria."
  )
  quit(save = "no", status = 0)
}

# ---------------------------------------------------------------------------
# Per-page plot
# ---------------------------------------------------------------------------

END_LEVELS <- c("5prime", "3prime")
END_LABELS <- c(`5prime` = "5′ end", `3prime` = "3′ end")

build_page <- function(key) {
  key_col <- if (is.character(key)) "species_name" else "taxid"

  lines <- list(); spans <- list(); plats <- list()

  for (i in seq_len(n_samp)) {
    prof <- profiles[[i]]; sdf <- sdfs[[i]]; gdf <- gdfs[[i]]
    for (e in END_LEVELS) {
      sub <- prof[prof[[key_col]] == key & prof$end == e & prof$position < args$max_pos, , drop = FALSE]
      if (nrow(sub) == 0L) next
      sub <- sub[order(sub$position), , drop = FALSE]

      ps <- args$plateau_start; pe <- args$plateau_end
      score <- NA_real_; n_reads <- 0L; pval <- NA_real_
      if (!is.null(sdf)) {
        row <- sdf[sdf[[key_col]] == key & sdf$end == e, , drop = FALSE]
        if (nrow(row) > 0L) {
          if (all(c("plateau_pos_start", "plateau_pos_end") %in% names(row))) {
            ps <- as.integer(row$plateau_pos_start[1]); pe <- as.integer(row$plateau_pos_end[1])
          }
          score <- row$damage_score[1]
          n_reads <- as.integer(row$n_reads[1])
          if ("damage_pvalue" %in% names(row)) pval <- row$damage_pvalue[1]
        }
      } else {
        g <- gdf[gdf[[key_col]] == key, , drop = FALSE]
        sc <- if (e == "5prime") "damage_score" else "damage_score_3prime"
        if (nrow(g) > 0L) { score <- g[[sc]][1]; n_reads <- as.integer(g$n_reads[1]) }
      }

      lbl <- sprintf("%s  (n=%s, Δ=%.2f%%", labels[i],
                     format(n_reads, big.mark = ","), score * 100)
      if (!is.na(pval)) lbl <- paste0(lbl, sprintf(", p=%.1e", pval))
      lbl <- paste0(lbl, ")")

      lines[[length(lines) + 1L]] <- data.frame(
        end = e, position = sub$position,
        pct = sub$frac_unclassified * 100,
        series = lbl, sample_i = i, stringsAsFactors = FALSE
      )
      spans[[length(spans) + 1L]] <- data.frame(
        end = e, xmin = ps, xmax = pe, sample_i = i, stringsAsFactors = FALSE
      )
      inwin <- sub$position >= ps & sub$position <= pe
      plats[[length(plats) + 1L]] <- data.frame(
        end = e,
        yint = mean(sub$frac_unclassified[inwin], na.rm = TRUE) * 100,
        series = lbl, sample_i = i, stringsAsFactors = FALSE
      )
    }
  }

  if (length(lines) == 0L) return(NULL)
  lines <- dplyr::bind_rows(lines)
  spans <- dplyr::bind_rows(spans)
  plats <- dplyr::bind_rows(plats)
  lines$end <- factor(lines$end, levels = END_LEVELS)
  spans$end <- factor(spans$end, levels = END_LEVELS)
  plats$end <- factor(plats$end, levels = END_LEVELS)

  series_levels <- unique(lines$series)
  pal <- colours[vapply(series_levels, function(s) lines$sample_i[match(s, lines$series)], integer(1))]
  lty <- ltypes[vapply(series_levels, function(s) lines$sample_i[match(s, lines$series)], integer(1))]
  names(pal) <- series_levels; names(lty) <- series_levels
  lines$series <- factor(lines$series, levels = series_levels)
  plats$series <- factor(plats$series, levels = series_levels)
  spans$series <- factor(lines$series[match(spans$sample_i, lines$sample_i)],
                         levels = series_levels)

  page_title <- if (is.character(key)) key else {
    nm <- ""
    for (g in gdfs) {
      m <- g[g$taxid == key, , drop = FALSE]
      if (nrow(m) > 0L && "species_name" %in% names(m)) { nm <- as.character(m$species_name[1]); break }
    }
    if (nzchar(nm)) sprintf("taxid %s — %s", key, nm) else paste("taxid", key)
  }

  ggplot(lines, aes(x = .data$position, y = .data$pct)) +
    geom_rect(data = spans, inherit.aes = FALSE,
              aes(xmin = .data$xmin, xmax = .data$xmax, ymin = -Inf, ymax = Inf,
                  fill = .data$series),
              alpha = 0.08, show.legend = FALSE) +
    geom_hline(data = plats, aes(yintercept = .data$yint, colour = .data$series),
               linetype = "dotted", linewidth = 0.35, alpha = 0.7,
               show.legend = FALSE) +
    geom_line(aes(colour = .data$series, linetype = .data$series), linewidth = 0.7) +
    geom_point(aes(colour = .data$series), size = 0.9, show.legend = FALSE) +
    scale_colour_manual(values = pal, name = NULL) +
    scale_fill_manual(values = pal, guide = "none") +
    scale_linetype_manual(values = lty, name = NULL) +
    facet_wrap(~ end, nrow = 1, labeller = labeller(end = END_LABELS)) +
    coord_cartesian(xlim = c(-0.5, args$max_pos - 0.5)) +
    expand_limits(y = 0) +
    labs(
      title = paste("aDNA damage profiles —", page_title),
      x = "Position from read end",
      y = "Unclassified k-mers (%)",
      caption = paste(
        "Rising tail beyond plateau reflects read-length artifact",
        "(short reads: distant positions approach the opposite end)."
      )
    ) +
    theme_screen() +
    theme(legend.position = "top", legend.direction = "vertical")
}

# ---------------------------------------------------------------------------
# Write multi-page PDF
# ---------------------------------------------------------------------------

open_pdf_device(args$output, width = 11, height = 4.4)

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
      annotate("text", x = 0.5, y = 0.6, label = "No damage-profile plots were generated",
               size = 4.6, fontface = "bold") +
      annotate("text", x = 0.5, y = 0.42,
               label = "Selected taxa had no matching rows in the profile table.",
               size = 3.4) +
      xlim(0, 1) + ylim(0, 1) + theme_void()
  )
  pages <- 1L
}
invisible(dev.off())
message("Saved ", args$output, " (", pages, " page(s))")
