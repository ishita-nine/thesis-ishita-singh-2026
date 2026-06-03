#!/usr/bin/env Rscript

# author: ishita singh
# project: senior thesis
# date: 4/26/26
# purpose: verify that duplicated sample_cell_id rows in the combined positive
#          pairs file represent legitimate 1h + multi-l b cells (one heavy
#          chain, multiple distinct light chains) and not pipeline bugs.
# input:  combined_test_pos_3_sources.tsv
# output: console report only — no file written.

library(data.table)

base_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab"
input_path <- file.path(base_dir, "outputs/combined_training_pos_3_sources.tsv")

dt <- fread(input_path)
message("loaded ", nrow(dt), " rows from ", input_path)

# count rows per (sample_cell_id, source) so we know which cells are duplicated
dt[, n_rows := .N, by = .(sample_cell_id, source)]
dups <- dt[n_rows > 1]
message("duplicated rows: ", nrow(dups), " (across ",
        uniqueN(dups[, .(sample_cell_id, source)]), " unique cells)")

# ---- check 1: every duplicated cell should have exactly 1 distinct heavy ----
# a clean 1h + multi-l cell has the same heavy chain repeated across rows.
# more than 1 distinct heavy under one sample_cell_id suggests a doublet or
# some other pipeline bug.
message("\n=== check 1: distinct heavies per duplicated cell ===")
heavy_check <- dups[, .(n_heavies = uniqueN(heavy)),
                    by = .(sample_cell_id, source)]
suspicious_heavies <- heavy_check[n_heavies > 1]
if (nrow(suspicious_heavies) == 0) {
  message("clean: every duplicated cell has exactly 1 distinct heavy chain.")
} else {
  message("suspicious: ", nrow(suspicious_heavies),
          " duplicated cells have multiple distinct heavies.")
  message("breakdown by source:")
  print(suspicious_heavies[, .N, by = source])
  message("first 10 examples:")
  print(head(suspicious_heavies, 10))
}

# ---- check 2: every duplicated cell should have >1 distinct light ----
# the whole point of a duplicated cell is that it has multiple lights paired
# with one heavy. if a cell is duplicated but has only 1 distinct light, the
# rows are exact copies — meaning the upstream join produced duplicate output
# for some reason (different bug than 1h + multi-l biology).
message("\n=== check 2: distinct lights per duplicated cell ===")
light_check <- dups[, .(n_lights = uniqueN(light)),
                    by = .(sample_cell_id, source)]
suspicious_lights <- light_check[n_lights == 1]
if (nrow(suspicious_lights) == 0) {
  message("clean: every duplicated cell has multiple distinct light chains.")
} else {
  message("suspicious: ", nrow(suspicious_lights),
          " duplicated cells have only 1 distinct light (exact-duplicate rows).")
  message("breakdown by source:")
  print(suspicious_lights[, .N, by = source])
  message("first 10 examples:")
  print(head(suspicious_lights, 10))
}

# ---- summary: distribution of (n_heavies, n_lights) across duplicates ----
# this gives a global picture of what duplicated cells look like.
# expected: most rows in a single bucket (1 heavy, 2+ lights).
message("\n=== summary: shape of duplicated cells ===")
shape <- merge(heavy_check, light_check,
               by = c("sample_cell_id", "source"))
print(shape[, .N, by = .(source, n_heavies, n_lights)][order(source, -N)])

message("\ndone.")