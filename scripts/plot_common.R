#!/usr/bin/env Rscript
# ---------------------------------------------------------------------------
# plot_common.R
#
# Shared helpers for the damage plotting scripts: CLI parsing, hit-species
# selection, and placeholder plots.
#
# The CLI parser is deliberately hand-rolled. The existing Python entry points
# use argparse with nargs="+" (repeatable multi-value flags), and the workflow
# calls them with exactly those signatures; R's `argparse` package would
# reproduce that but shells out to Python on every invocation, and `optparse`
# does not support multi-value flags at all. Roughly 40 lines of base R keeps
# the CLI contract identical with no dependency and no subprocess.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(ggplot2)
  library(readr)
  library(dplyr)
})

# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

#' Parse `--flag value [value ...]` command lines.
#'
#' spec is a named list; each element is list(type=, nargs=, default=).
#'   type  : "character" | "integer" | "double" | "flag"
#'   nargs : 1 or "+"   ("+" collects values until the next --flag)
#' Flag names use dashes on the command line and underscores in the result,
#' matching the argparse convention the Python scripts followed.
parse_cli <- function(args, spec) {
  out <- lapply(spec, function(s) s$default)
  names(out) <- names(spec)

  cli_name <- function(nm) paste0("--", gsub("_", "-", nm))
  lookup <- setNames(names(spec), vapply(names(spec), cli_name, character(1)))

  i <- 1L
  while (i <= length(args)) {
    tok <- args[[i]]
    if (!startsWith(tok, "--")) {
      stop("unexpected positional argument: ", tok, call. = FALSE)
    }
    key <- lookup[[tok]]
    if (is.null(key) || is.na(key)) {
      stop("unknown option: ", tok, call. = FALSE)
    }
    s <- spec[[key]]

    if (identical(s$type, "flag")) {
      out[[key]] <- TRUE
      i <- i + 1L
      next
    }

    vals <- character(0)
    j <- i + 1L
    while (j <= length(args) && !startsWith(args[[j]], "--")) {
      vals <- c(vals, args[[j]])
      j <- j + 1L
    }
    if (length(vals) == 0L) {
      stop("option ", tok, " expects a value", call. = FALSE)
    }
    if (identical(s$nargs, 1) && length(vals) > 1L) {
      stop("option ", tok, " expects a single value", call. = FALSE)
    }

    out[[key]] <- switch(
      s$type,
      integer   = as.integer(vals),
      double    = as.numeric(vals),
      character = vals
    )
    i <- j
  }
  out
}

# ---------------------------------------------------------------------------
# Hit-species selection  (ported from the former hit_species_selection.py)
# ---------------------------------------------------------------------------

#' Select species names for one sample from the integrated summary table.
#'
#' Selection order, matching the Python implementation exactly:
#'   1) filter to sample_id
#'   2) keep rows whose hit_criteria_flag contains ALL required tokens
#'      (empty token list = no filtering)
#'   3) de-duplicate by species_name
#'   4) rank by lowest damage_pvalue, then species_name
#'   5) cap at max_keys when max_keys > 0
#'
#' sample_id is read as character throughout: an all-digit id such as 018345
#' would otherwise lose its leading zero and match nothing.
select_hit_species <- function(hits_path, sample_id,
                               required_flag_tokens = character(0),
                               max_keys = 200L) {
  empty <- list(species = character(0), n_truncated = 0L)
  if (is.null(hits_path) || is.null(sample_id)) return(empty)

  hdf <- tryCatch(
    suppressWarnings(readr::read_tsv(
      hits_path,
      col_types = readr::cols(
        sample_id        = readr::col_character(),
        species_name     = readr::col_character(),
        hit_criteria_flag = readr::col_character(),
        damage_pvalue    = readr::col_double(),
        .default         = readr::col_skip()
      ),
      progress = FALSE
    )),
    error = function(e) {
      message("WARNING: could not read --hits file (", hits_path, "): ", conditionMessage(e))
      NULL
    }
  )
  if (is.null(hdf) || nrow(hdf) == 0L) return(empty)
  if (!"species_name" %in% names(hdf)) return(empty)

  hdf <- hdf[!is.na(hdf$sample_id) & as.character(hdf$sample_id) == as.character(sample_id), , drop = FALSE]
  if (nrow(hdf) == 0L) return(empty)

  if (length(required_flag_tokens) > 0L && "hit_criteria_flag" %in% names(hdf)) {
    flags <- ifelse(is.na(hdf$hit_criteria_flag), "", hdf$hit_criteria_flag)
    tokens <- strsplit(flags, ";", fixed = TRUE)
    keep <- vapply(
      tokens,
      function(tk) all(required_flag_tokens %in% tk),
      logical(1)
    )
    hdf <- hdf[keep, , drop = FALSE]
  }
  if (nrow(hdf) == 0L) return(empty)

  hdf <- hdf[!is.na(hdf$species_name) & nzchar(hdf$species_name), , drop = FALSE]
  if (nrow(hdf) == 0L) return(empty)

  pv <- if ("damage_pvalue" %in% names(hdf)) hdf$damage_pvalue else rep(NA_real_, nrow(hdf))
  best <- data.frame(
    species_name = hdf$species_name,
    pvalue = pv,
    stringsAsFactors = FALSE
  )
  best <- best[order(best$species_name, best$pvalue, na.last = TRUE), , drop = FALSE]
  best <- best[!duplicated(best$species_name), , drop = FALSE]
  best <- best[order(best$pvalue, best$species_name, na.last = TRUE), , drop = FALSE]

  species <- best$species_name
  n_truncated <- 0L
  if (!is.null(max_keys) && length(max_keys) == 1L && !is.na(max_keys) && max_keys > 0L &&
      length(species) > max_keys) {
    n_truncated <- length(species) - as.integer(max_keys)
    species <- species[seq_len(max_keys)]
  }
  list(species = species, n_truncated = n_truncated)
}

resolve_required_hit_flags <- function(flags) {
  if (length(flags) > 0L) return(flags)
  c("damage_rate", "within_genus_relative_abundance", "classified_rate")
}

# ---------------------------------------------------------------------------
# Placeholder output
# ---------------------------------------------------------------------------

#' PDF device that can encode the UTF-8 used in axis labels (primes, minus
#' signs, en dashes). The default pdf() device substitutes "-" for U+2212 and
#' warns; cairo_pdf renders them correctly.
pdf_device <- function() {
  if (isTRUE(capabilities("cairo"))) grDevices::cairo_pdf else "pdf"
}

#' Open a multi-page PDF device, preferring cairo_pdf for UTF-8 support.
open_pdf_device <- function(path, width, height) {
  dev <- pdf_device()
  if (is.function(dev)) {
    dev(path, width = width, height = height, onefile = TRUE)
  } else {
    grDevices::pdf(path, width = width, height = height, onefile = TRUE)
  }
}

#' Write a placeholder PDF so Snakemake outputs exist even with nothing to plot.
write_empty_plot <- function(output_path, heading, note, width = 9, height = 4.5) {
  p <- ggplot() +
    annotate("text", x = 0.5, y = 0.62, label = heading,
             size = 4.6, fontface = "bold") +
    annotate("text", x = 0.5, y = 0.42, label = note, size = 3.4) +
    xlim(0, 1) + ylim(0, 1) +
    theme_void()
  ggsave(output_path, p, width = width, height = height, device = pdf_device())
  message("Saved ", output_path, " (empty)")
}

theme_screen <- function(base_size = 9) {
  theme_classic(base_size = base_size) +
    theme(
      plot.title    = element_text(size = base_size + 1, hjust = 0.5),
      plot.subtitle = element_text(size = base_size - 1, hjust = 0.5),
      plot.caption  = element_text(size = base_size - 1, colour = "grey40",
                                   face = "italic", hjust = 0.5),
      axis.text     = element_text(size = base_size - 1),
      strip.background = element_blank(),
      strip.text    = element_text(size = base_size, face = "bold"),
      legend.key.size = unit(0.8, "lines"),
      legend.text   = element_text(size = base_size - 2),
      legend.title  = element_text(size = base_size - 1)
    )
}
