#!/usr/bin/env Rscript

# Author: Ishita Singh
# Project: Senior Thesis
# Data: 3/2/26
# Purpose: Generate negative test pairs as in Immunomatch (shuffle method, based on CDR3 length), given positive pairs
# Location: To be run from thesis-venv in Discovery scratch, data in lab directory 

library(data.table)
library(parallel)

# discovery location
out_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/outputs"

# mounted location 
# out_dir <- "/Volumes/HoehnK/Ishita/immunomatch-lab/jaffe/outputs"

ncores <- as.integer(Sys.getenv("SLURM_CPUS_PER_TASK","3"))

# load positives
# positive_pairs <- fread(file.path(out_dir, "jaffe_test_positive.tsv"), drop = "vgl")
positive_pairs <- fread(file.path(out_dir, "combined_training_pos_3_sources.tsv"), drop = "vgl")

# useful for the combined case, we dont want to carry over source to negative pairs 
if ("source" %in% names(positive_pairs)) {
  positive_pairs[, source := NULL]
}

# add cdrl3 length column 
positive_pairs[, cdrl3_length := nchar(cdr3.light)]

# assign unique numeric IDs for each heavy and light sequence
positive_pairs[, heavy_id := .GRP, by = heavy]
positive_pairs[, light_id := .GRP, by = light]

# precompute positive pair integer hash for fast lookup
max_light <- max(positive_pairs$light_id) + 1

positive_keys <- unique(positive_pairs[,heavy_id * max_light + light_id])
global_positive_hash <- setNames(rep(TRUE,length(positive_keys)),as.character(positive_keys))

set.seed(9)

length_groups <- split(positive_pairs, by = "cdrl3_length")

# precompute light chain columns
light_cols <- grep("\\.light$", names(positive_pairs), value = TRUE)

swap_group <- function(group, light_cols, positive_hash, max_light) {
  n_group <- nrow(group)
  if (n_group < 2) return(NULL)
  
  # precompute light chain values
  light_matrix <- as.matrix(group[, ..light_cols])
  
  neg_rows <- vector("list", n_group)
  pos_rows <- vector("list", n_group)
  neg_idx <- 0
  pos_idx <- 0

  order_i <- sample(seq_len(n_group))
  
  for (idx in seq_along(order_i)) {
    i <- order_i[idx]
    
    success <- FALSE
    attempts <- 0
    
    row_i <- group[i]
    
    while (!success && attempts < 250) {
      attempts <- attempts + 1
      
      j <- sample.int(n_group, 1)
      if (j == i) next  # skip self
      
      row_j <- group[j]
      
      # candidate negative check using IDs
      neg_key <- row_i$heavy_id * max_light + row_j$light_id
      if (!is.na(positive_hash[as.character(neg_key)])) next  # skip if positive exists
      
      # fill row with heavy (i) and light values
      neg_row <- copy(row_i)
      neg_row[, (light_cols) := as.list(light_matrix[j, ])]
      neg_row[, sample_cell_id := paste0(row_i$sample_cell_id, "-", row_j$sample_cell_id)]
      neg_row[, heavy := row_i$heavy]
      neg_row[, light := row_j$light]
      neg_row[, heavy_id := row_i$heavy_id]
      neg_row[, light_id := row_j$light_id]
      
      # store negative and original positive
      neg_idx <- neg_idx + 1
      neg_rows[[neg_idx]] <- neg_row
      pos_idx <- pos_idx + 1
      pos_rows[[pos_idx]] <- row_i
      
      success <- TRUE
    }
    
    if (idx %% 500 == 0) message("Processing idx ", idx, " of ", n_group, " (CDRL3 length: ", group$cdrl3_length[1], ")")
  }
  
  list(
    neg = if (neg_idx > 0) rbindlist(neg_rows[1:neg_idx]) else NULL,
    pos = if (pos_idx > 0) rbindlist(pos_rows[1:pos_idx]) else NULL
  )
}

# run swap group on each group (grouped by CDRL3 length)
results <- mclapply(length_groups,
                    function(g) {
                      tryCatch(
                        swap_group(g, light_cols, global_positive_hash, max_light),
                        error = function(e) { message("Error in group with CDRL3 length ", g$cdrl3_length[1], ": ", e); NULL }
                      )
                    },
                    mc.cores = ncores,mc.preschedule = FALSE, mc.set.seed = TRUE)
results <- results[sapply(results, function(x) !is.null(x) && !is.null(x$neg) && !is.null(x$pos))]

negative_pairs <- rbindlist(lapply(results, function(x) x$neg))
positive_used <- rbindlist(lapply(results, function(x) x$pos))

# checking that the number of positive pairs left and negative pairs generated are the same 
message("Negatives: ", nrow(negative_pairs))
message("Positives: ", nrow(positive_used))
stopifnot(nrow(negative_pairs) == nrow(positive_used))

setcolorder(negative_pairs, names(positive_pairs))

# checking that there is no overlap in positives and negatives 
positive_seqs <- paste0(positive_used$heavy, "_", positive_used$light)
negative_seqs <- paste0(negative_pairs$heavy, "_", negative_pairs$light)
message(sum(negative_seqs %in% positive_seqs))

# fwrite(negative_pairs, file.path(out_dir, "jaffe_test_negative_cdrl3_shuffle.tsv"), sep = "\t")
# fwrite(positive_used, file.path(out_dir, "jaffe_test_positive_cdrl3_shuffle.tsv"), sep = "\t")

fwrite(negative_pairs, file.path(out_dir, "combined_training_neg_cdrl3_shuffle.tsv"), sep = "\t")
fwrite(positive_used, file.path(out_dir, "combined_training_pos_used_cdrl3_shuffle.tsv"), sep = "\t")

message("Done")