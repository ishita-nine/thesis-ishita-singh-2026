#!/usr/bin/env Rscript

# author: ishita singh
# project: senior thesis
# date: 4/26/26
# purpose: combine positive pairs from three datasets (cell, joi, jaffe) into
#          one minimal-schema tsv with a source label per row. output feeds
#          into the existing v-gene negative-pair generation script.
# location: to be run from thesis-venv in discovery scratch, data in lab dir

library(data.table)

base_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab"

input_files <- list(
  cell  = file.path(base_dir, "outputs/test_positive.tsv"),
  joi   = file.path(base_dir, "outputs/training_positive_20k.tsv"),
  jaffe = file.path(base_dir, "jaffe/outputs/jaffe_training_positive.tsv")
)

out_path <- file.path(base_dir, "outputs/combined_training_pos_3_sources.tsv")

keep_cols <- c("sample_cell_id", "vgh", "vgl", "heavy", "light",
               "sequence_alignment.heavy", "sequence_alignment.light",
               "c_call.heavy", "c_call.light",
               "locus.light", "cdr3.light")

# helper: read one positive-pairs tsv, validate, reduce to minimal schema
load_positives <- function(path, source_label) {
  if (!file.exists(path)) {
    stop("input file not found: ", path)
  }
  dt <- fread(path)
  missing <- setdiff(keep_cols, names(dt))
  if (length(missing) > 0) {
    stop("file ", path, " is missing required columns: ",
         paste(missing, collapse = ", "))
  }
  dt <- dt[, ..keep_cols]
  dt[, source := source_label]
  message("loaded ", nrow(dt), " pairs from ", source_label, " (", path, ")")
  dt
}

# load all three datasets with the requested source labels
pos_list <- list(
  load_positives(input_files$cell,  "cell"),
  load_positives(input_files$joi,   "joi"),
  load_positives(input_files$jaffe, "jaffe")
)

# combine
positive_pairs <- rbindlist(pos_list, use.names = TRUE)
message("combined total: ", nrow(positive_pairs), " pairs")

# drop rows where vgl prefix disagrees with locus.light
# only from joi where igblast assigned kappa v-genes to lambda-locus sequence
# high percentage of Xs
n_before <- nrow(positive_pairs)
positive_pairs <- positive_pairs[substr(vgl, 1, 3) == locus.light]
message("dropped ", n_before - nrow(positive_pairs), 
        " v/locus mismatched rows")

# sanity check: sample_cell_id should be unique across the combined pool.
# each upstream pipeline builds it with sample/donor prefixes, so cross-dataset
# collisions should be impossible. fail loudly if that assumption breaks.
n_unique <- uniqueN(positive_pairs$sample_cell_id)
if (n_unique != nrow(positive_pairs)) {
  warning("sample_cell_id is not unique across combined datasets: ",
          n_unique, " unique ids vs ", nrow(positive_pairs), " rows. ",
          "this could cause issues in downstream joins.")
} else {
  message("all sample_cell_id values are unique across the combined pool.")
}

# distribution by source for the log
message("breakdown by source:")
print(positive_pairs[, .N, by = source])

# write output as tsv
dir.create(dirname(out_path), showWarnings = FALSE, recursive = TRUE)
fwrite(positive_pairs, out_path, sep = "\t")
message("wrote combined positives to: ", out_path)