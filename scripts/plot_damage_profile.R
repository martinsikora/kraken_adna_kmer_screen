#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# plot_damage_profile.R
#
# Absolute-position aDNA damage profiles from workflow damage TSV outputs.
#
#   X axis : position from read end (0 = terminal k-mer)
#   Y axis : fraction of unclassified k-mers (%)
#
# One page per taxon. Columns are the read ends, with the 3' column mirrored so
# its terminus sits on the right, following mapDamage. Rows are: the pooled
# profile, carrying the plateau window and damage estimates, then one row per
# read-length stratum.
#
# The strata matter because a read reaches k-mer position j only if its length
# is at least k + j. A pooled profile therefore changes its read composition
# along the x axis -- far positions are computed from progressively fewer, and
# longer, reads, which carry systematically higher unclassified rates. Point
# size shows how many reads are behind each position so that attrition is
# visible rather than implicit.
#
# Line colour follows the mapDamage convention: 5' red, 3' blue. Several
# samples can be overlaid by passing --profile/--global/--stats more than once;
# sample identity is then carried by linetype.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
  library(scales)
})

script_dir <- function() {
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f)) dirname(normalizePath(f)) else "."
}
source(file.path(script_dir(), "plot_common.R"))

SAMPLE_LINETYPES <- c("solid", "dashed", "dotdash", "dotted")
END_LEVELS <- c("5prime", "3prime")
END_LABELS <- c(`5prime` = "5′ end", `3prime` = "3′ end")
END_PAL    <- c(`5′ end` = "#d6604d", `3′ end` = "#2166ac")
POOLED_LAB <- "all reads"

SPEC <- list(
  profile             = list(type = "character", nargs = "+", default = character(0)),
  profile_stratified  = list(type = "character", nargs = "+", default = character(0)),
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
  max_pos             = list(type = "integer",   nargs = 1,   default = 10L),
  plateau_start       = list(type = "integer",   nargs = 1,   default = 3L),
  plateau_end         = list(type = "integer",   nargs = 1,   default = 9L),
  pvalue_threshold    = list(type = "double",    nargs = 1,   default = 0.05),
  kmer_size           = list(type = "integer",   nargs = 1,   default = NULL)
)

args <- parse_cli(commandArgs(trailingOnly = TRUE), SPEC)
stopifnot(length(args$profile) > 0, !is.null(args$output))
if (length(args$profile) != length(args$global)) {
  stop("--profile and --global must match in count", call. = FALSE)
}
if (length(args$stats) > 0 && length(args$stats) != length(args$profile)) {
  stop("--stats count must match --profile count", call. = FALSE)
}
if (length(args$profile_stratified) > 0 &&
    length(args$profile_stratified) != length(args$profile)) {
  stop("--profile-stratified count must match --profile count", call. = FALSE)
}

n_samp <- length(args$profile)
labels <- if (length(args$label) == n_samp) args$label else paste("Sample", seq_len(n_samp))
ltypes <- SAMPLE_LINETYPES[(seq_len(n_samp) - 1) %% length(SAMPLE_LINETYPES) + 1]

read_any <- function(p) suppressWarnings(readr::read_tsv(
  p, col_types = readr::cols(.default = readr::col_guess(),
                             species_name = readr::col_character()),
  progress = FALSE))

profiles <- lapply(args$profile, read_any)
gdfs     <- lapply(args$global, read_any)
sdfs     <- if (length(args$stats) > 0) lapply(args$stats, read_any) else vector("list", n_samp)
strats   <- if (length(args$profile_stratified) > 0) {
  lapply(args$profile_stratified, read_any)
} else vector("list", n_samp)

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
} else {
  keys <- resolve_keys(gdfs, args$taxids, args$species, args$pvalue_threshold)
  if (length(args$taxids) == 0 && length(args$species) == 0 &&
      args$max_keys > 0 && length(keys) > args$max_keys) {
    message("WARNING: auto-selected taxa truncated to ", args$max_keys,
            " (dropped ", length(keys) - args$max_keys, " by --max-keys).")
    keys <- keys[seq_len(args$max_keys)]
  }
}

if (length(keys) > 0 && is.character(keys)) {
  available <- unique(unlist(lapply(profiles, function(p)
    if ("species_name" %in% names(p)) stats::na.omit(as.character(p$species_name))
    else character(0))))
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
  write_empty_plot(args$output, "No damage-profile plots were generated",
                   "No taxa were selected from --hits or auto-selection criteria.")
  quit(save = "no", status = 0)
}

# k-mer size: recorded per row by screen_unit (k = read length - n_kmers + 1),
# overridable for profile tables written before it was carried. Without it the
# axis shows k-mer indices only, since the bases a k-mer spans are unknown.
KMER <- if (!is.null(args$kmer_size)) args$kmer_size else {
  ks <- unique(unlist(lapply(profiles, function(d)
    if ("kmer_size" %in% names(d)) d$kmer_size else integer(0))))
  ks <- ks[!is.na(ks) & ks > 0]
  if (length(ks) == 1L) as.integer(ks) else {
    if (length(ks) > 1L) message("WARNING: profile tables disagree on k-mer size (",
                                 paste(ks, collapse = ", "), "); labelling indices only.")
    NA_integer_
  }
}

# k-mer index j spans read bases j+1 .. j+k counted from that end, so the same
# formula labels both the 5' and the mirrored 3' axis.
index_labels <- function(v) {
  j <- abs(v)
  if (is.na(KMER)) return(as.character(j))
  paste0(j, "\n", j + 1, "–", j + KMER)
}

# strata present in the data, ordered by lower bound
strata_levels <- unique(unlist(lapply(strats, function(d)
  if (!is.null(d) && "stratum" %in% names(d)) as.character(d$stratum) else character(0))))
if (length(strata_levels) > 0) {
  strata_levels <- strata_levels[order(as.numeric(sub("-.*", "", strata_levels)))]
}
panel_levels <- c(POOLED_LAB, strata_levels)

# ---------------------------------------------------------------------------
# Per-page plot
# ---------------------------------------------------------------------------

build_page <- function(key) {
  key_col <- if (is.character(key)) "species_name" else "taxid"
  rows <- list(); spans <- list(); plats <- list(); ann <- list()

  for (i in seq_len(n_samp)) {
    prof <- profiles[[i]]; sdf <- sdfs[[i]]; gdf <- gdfs[[i]]; st <- strats[[i]]

    for (e in END_LEVELS) {
      sub <- prof[prof[[key_col]] == key & prof$end == e &
                    prof$position < args$max_pos, , drop = FALSE]
      if (nrow(sub) == 0L) next
      sub <- sub[order(sub$position), , drop = FALSE]

      ps <- args$plateau_start; pe <- args$plateau_end
      score <- NA_real_; n_reads <- 0L; pval <- NA_real_
      if (!is.null(sdf)) {
        row <- sdf[sdf[[key_col]] == key & sdf$end == e, , drop = FALSE]
        if (nrow(row) > 0L) {
          if (all(c("plateau_pos_start", "plateau_pos_end") %in% names(row))) {
            ps <- as.integer(row$plateau_pos_start[1])
            pe <- as.integer(row$plateau_pos_end[1])
          }
          score <- row$damage_score[1]; n_reads <- as.integer(row$n_reads[1])
          if ("damage_pvalue" %in% names(row)) pval <- row$damage_pvalue[1]
        }
      } else {
        g <- gdf[gdf[[key_col]] == key, , drop = FALSE]
        sc <- if (e == "5prime") "damage_score" else "damage_score_3prime"
        if (nrow(g) > 0L) { score <- g[[sc]][1]; n_reads <- as.integer(g$n_reads[1]) }
      }

      rows[[length(rows) + 1L]] <- data.frame(
        panel = POOLED_LAB, end = e, position = sub$position,
        pct = sub$frac_unclassified * 100, n_kmers = sub$n_kmers,
        sample_i = i, stringsAsFactors = FALSE)
      spans[[length(spans) + 1L]] <- data.frame(
        panel = POOLED_LAB, end = e, ps = ps, pe = pe,
        sample_i = i, stringsAsFactors = FALSE)
      inwin <- sub$position >= ps & sub$position <= pe
      plats[[length(plats) + 1L]] <- data.frame(
        panel = POOLED_LAB, end = e,
        yint = mean(sub$frac_unclassified[inwin], na.rm = TRUE) * 100,
        sample_i = i, stringsAsFactors = FALSE)

      lbl <- sprintf("%s  (n=%s, Δ=%.2f%%", labels[i],
                     format(n_reads, big.mark = ","), score * 100)
      if (!is.na(pval)) lbl <- paste0(lbl, sprintf(", p=%.1e", pval))
      ann[[length(ann) + 1L]] <- data.frame(
        panel = POOLED_LAB, end = e, label = paste0(lbl, ")"),
        sample_i = i, stringsAsFactors = FALSE)

      if (!is.null(st) && "stratum" %in% names(st)) {
        ss <- st[st[[key_col]] == key & st$end == e &
                   st$position < args$max_pos, , drop = FALSE]
        if (nrow(ss) > 0L) {
          rows[[length(rows) + 1L]] <- data.frame(
            panel = as.character(ss$stratum), end = e, position = ss$position,
            pct = ss$frac_unclassified * 100, n_kmers = ss$n_kmers,
            sample_i = i, stringsAsFactors = FALSE)
        }
      }
    }
  }

  if (length(rows) == 0L) return(NULL)
  dat   <- dplyr::bind_rows(rows)
  spans <- dplyr::bind_rows(spans)
  plats <- dplyr::bind_rows(plats)
  ann   <- dplyr::bind_rows(ann)

  present <- panel_levels[panel_levels %in% dat$panel]
  strip <- vapply(present, function(p)
    if (identical(p, POOLED_LAB)) POOLED_LAB else paste0(p, " bp"), character(1))

  fac <- function(d) {
    d$panel <- factor(d$panel, levels = present, labels = strip)
    d$end   <- factor(d$end, levels = END_LEVELS, labels = END_LABELS)
    d[!is.na(d$panel), , drop = FALSE]
  }
  dat <- fac(dat); spans <- fac(spans); plats <- fac(plats); ann <- fac(ann)

  # mapDamage orientation: mirror the 3' column so its terminus is on the right.
  # ggplot cannot reverse a scale for one facet only, so 3' x is negated and the
  # axis labels take the absolute value.
  three <- END_LABELS[["3prime"]]
  dat$x    <- ifelse(dat$end == three, -dat$position, dat$position)
  spans$x1 <- ifelse(spans$end == three, -spans$ps, spans$ps)
  spans$x2 <- ifelse(spans$end == three, -spans$pe, spans$pe)
  ann$x    <- ifelse(ann$end == three, Inf, -Inf)
  ann$h    <- ifelse(ann$end == three, 1.03, -0.03)

  dat$sample <- factor(labels[dat$sample_i], levels = labels[sort(unique(dat$sample_i))])
  lty <- ltypes[sort(unique(dat$sample_i))]; names(lty) <- levels(dat$sample)

  page_title <- if (is.character(key)) key else {
    nm <- ""
    for (g in gdfs) {
      m <- g[g$taxid == key, , drop = FALSE]
      if (nrow(m) > 0L && "species_name" %in% names(m)) {
        nm <- as.character(m$species_name[1]); break
      }
    }
    if (nzchar(nm)) sprintf("taxid %s — %s", key, nm) else paste("taxid", key)
  }

  ggplot(dat, aes(.data$x, .data$pct, colour = .data$end)) +
    geom_rect(data = spans, inherit.aes = FALSE,
              aes(xmin = pmin(.data$x1, .data$x2), xmax = pmax(.data$x1, .data$x2),
                  ymin = -Inf, ymax = Inf, fill = .data$end),
              alpha = 0.08, show.legend = FALSE) +
    geom_hline(data = plats, aes(yintercept = .data$yint, colour = .data$end),
               linetype = "dotted", linewidth = 0.35, alpha = 0.7, show.legend = FALSE) +
    geom_line(aes(linetype = .data$sample), linewidth = 0.6) +
    geom_point(aes(size = .data$n_kmers), alpha = 0.9) +
    geom_text(data = ann, inherit.aes = FALSE,
              aes(x = .data$x, y = Inf, label = .data$label, colour = .data$end,
                  hjust = .data$h),
              vjust = 1.4, size = 2.1, show.legend = FALSE) +
    scale_colour_manual(values = END_PAL, guide = "none") +
    scale_fill_manual(values = END_PAL, guide = "none") +
    scale_linetype_manual(values = lty, name = NULL,
                          guide = if (n_samp > 1L) "legend" else "none") +
    scale_size_area(max_size = 2.6, name = "reads at\nthis index",
                    labels = scales::comma) +
    scale_x_continuous(breaks = scales::pretty_breaks(6), labels = index_labels) +
    facet_grid(panel ~ end, scales = "free", switch = "y") +
    labs(title = paste("aDNA damage profiles —", page_title),
         x = if (is.na(KMER)) "Position from read end (k-mer index)"
             else sprintf("Position from read end — k-mer index / read bases spanned (k=%d)", KMER),
         y = "Unclassified k-mers (%)",
         caption = paste(
           "Top row pooled, with the plateau window shaded and damage estimates",
           "annotated; lower rows split by read length.\nPoint size = reads",
           "contributing at that index. The 3′ axis is mirrored so both termini",
           "face outward.")) +
    theme_screen() +
    # full box around every facet: theme_classic() (via theme_screen) draws only
    # the left and bottom axis lines, which reads poorly on a facet grid
    theme(legend.position = "right", legend.direction = "vertical",
          panel.spacing = unit(0.5, "lines"), strip.placement = "outside",
          strip.text.y.left = element_text(angle = 0),
          plot.caption = element_text(size = 6.5),
          axis.text.x = element_text(size = 5.6, lineheight = 0.9),
          panel.border = element_rect(colour = "grey30", fill = NA, linewidth = 0.4),
          axis.line = element_blank())
}

# ---------------------------------------------------------------------------
# Write multi-page PDF
# ---------------------------------------------------------------------------

open_pdf_device(args$output, width = 9,
                height = max(4.5, 1.9 * max(1L, length(panel_levels)) + 1.4))

pages <- 0L
for (k in keys) {
  p <- build_page(k)
  if (is.null(p)) next
  print(p)
  pages <- pages + 1L
}
if (pages == 0L) {
  print(ggplot() +
    annotate("text", x = 0.5, y = 0.6, label = "No damage-profile plots were generated",
             size = 4.6, fontface = "bold") +
    annotate("text", x = 0.5, y = 0.42,
             label = "Selected taxa had no matching rows in the profile table.",
             size = 3.4) +
    xlim(0, 1) + ylim(0, 1) + theme_void())
  pages <- 1L
}
invisible(dev.off())
message("Saved ", args$output, " (", pages, " page(s))")
