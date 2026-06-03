#!/usr/bin/env Rscript

# Author: Ishita Singh
# Project: Senior Thesis
# Data: 3/2/26
# Purpose: Use data.table, instead of tidyverse, to generate negative test pairs (V-gene mismatched), given positive pairs
# Location: To be run from thesis-venv in Discovery scratch, data in lab directory 

library(data.table)

out_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/outputs"

positive_pairs <- fread(file.path(out_dir, "combined_training_pos_3_sources.tsv"))

# for combined pos: dropping source, treating as one pool
positive_pairs[, source := NULL]

n_pos <- nrow(positive_pairs)

observed_v_pairs <- unique(positive_pairs[, .(vgh, vgl)])

unique_vgh <- unique(positive_pairs$vgh)
unique_vgl <- unique(positive_pairs$vgl)

all_v_pairs <- CJ(vgh = unique_vgh, vgl = unique_vgl)

negative_v_pairs <- all_v_pairs[!observed_v_pairs,on = .(vgh, vgl)]

print(unique_vgh)
print(unique_vgl)
print(nrow(all_v_pairs))
print(nrow(observed_v_pairs))
print(nrow(negative_v_pairs))

# stopifnot(nrow(negative_v_pairs) > 0)
# 
# heavy_cols <- grep("\\.heavy$", names(positive_pairs), value = TRUE)
# light_cols <- grep("\\.light$", names(positive_pairs), value = TRUE)
# 
# h <- positive_pairs[, c(heavy_cols, "vgh", "heavy", "sample_cell_id"), with = FALSE]
# l <- positive_pairs[, c(light_cols, "vgl", "light", "sample_cell_id"), with = FALSE]
# 
# # setting key/index to allow grouping by vgene 
# setkey(h, vgh)
# setkey(l, vgl)
# 
# set.seed(9)
# 
# # sampling v gene pairs to use for negatives, with replacement 
# sampled_vs <- negative_v_pairs[sample(.N, n_pos, replace = TRUE)]
# 
# negative_list <- vector("list", n_pos)
# 
# for (i in seq_len(n_pos)) {
#   
#   vgh_i <- sampled_vs$vgh[i]
#   vgl_i <- sampled_vs$vgl[i]
#   
#   h_group <- h[.(vgh_i)]
#   l_group <- l[.(vgl_i)]
#   
#   heavy_row <- h_group[sample(.N, 1)]
#   light_row <- l_group[sample(.N, 1)]
#   
#   # grab ids first
#   heavy_id <- heavy_row$sample_cell_id
#   light_id <- light_row$sample_cell_id
#   
#   # remove sample_cell_id before binding
#   heavy_row[, sample_cell_id := NULL]
#   light_row[, sample_cell_id := NULL]
#   
#   # combine
#   neg_row <- cbind(heavy_row, light_row)
#   
#   neg_row[, vgh := vgh_i]
#   neg_row[, vgl := vgl_i]
#   
#   # reconstruct combined id
#   neg_row[, sample_cell_id := paste0(heavy_id, "_", light_id)]
#   
#   setcolorder(neg_row, names(positive_pairs))
#   
#   negative_list[[i]] <- neg_row
#   
#   if (i %% 100000 == 0) {
#     message("Generated ", i, " / ", n_pos, " negative pairs")
#   }
# }
# 
# negative_pairs <- rbindlist(negative_list)
# 
# stopifnot(identical(names(negative_pairs), names(positive_pairs)))
# 
# fwrite(
#   negative_pairs,
#   file.path(out_dir, "combined_training_neg_3_sources_vgene.tsv"), sep="\t"
# )
# 
# message("Done. Generated ", nrow(negative_pairs), " negative pairs.")