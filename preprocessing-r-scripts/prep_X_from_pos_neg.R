#!/usr/bin/env Rscript

# Author: Ishita Singh 
# Date: 1/27/25
# Term: 26W
# Project: Senior Thesis
# Purpose: Prep data for test my model, read in existing positive and negative pair data
# Location: To be run from thesis-venv in Discovery scratch, data in lab directory 
# CANNOT be run localy due to large file size 

suppressPackageStartupMessages({
  library(dplyr)
  library(stringr)
  library(readr)
  library(future)
  library(future.apply)
})

plan(multisession, workers = 7)

out_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs"

positive_file <- file.path(out_dir, "jaffe_training_positive.tsv")
negative_file <- file.path(out_dir, "jaffe_training_negative_vgene.tsv")

message("Reading input files...")
positive_pairs <- read_tsv(positive_file, show_col_types = FALSE)
negative_pairs <- read_tsv(negative_file, show_col_types = FALSE)

message("Positive rows: ", nrow(positive_pairs))
message("Negative rows: ", nrow(negative_pairs))

# helper function to replace, count x 
replace_X <- function(aa, nt) {
  aa <- strsplit(aa, "")[[1]]
  nt <- strsplit(nt, "")[[1]]
  
  for (i in seq_along(aa)) {
    if (aa[i] == "X") {
      codon_start <- (i - 1) * 3 + 1
      codon_end   <- codon_start + 2
      
      if (codon_end <= length(nt)) {
        codon <- nt[codon_start:codon_end]
        if (all(codon == ".")) {
          aa[i] <- "."
        }
      }
    }
  }
  paste0(aa, collapse = "")
}

count_X <- function(seq) str_count(seq, "X")

# using shared function for postive and light pairs 
# process_pairs <- function(df, tag = "") {
#   message("Processing ", tag, " pairs...")
#   
#   df <- df %>%
#     rowwise() %>%
#     mutate(
#       X_heavy_before = count_X(heavy),
#       X_light_before = count_X(light),
#       
#       heavy = replace_X(heavy, sequence_alignment.heavy),
#       light = replace_X(light, sequence_alignment.light),
#       
#       X_heavy_after = count_X(heavy),
#       X_light_after = count_X(light)
#     ) %>%
#     ungroup()
#   
#   summary <- df %>%
#     summarise(
#       total_X_heavy_before = sum(X_heavy_before),
#       total_X_light_before = sum(X_light_before),
#       total_X_heavy_after  = sum(X_heavy_after),
#       total_X_light_after  = sum(X_light_after)
#     )
#   
#   message(tag, " summary:")
#   print(summary)
#   
#   df
# }

# parallel processing function 
# parallel processing function
process_pairs <- function(df, tag = "") {
  message("Processing ", tag, " pairs...")
  
  df$X_heavy_before <- str_count(df$heavy, "X")
  df$X_light_before <- str_count(df$light, "X")
  
  df$heavy <- future_mapply(
    replace_X,
    df$heavy,
    df$sequence_alignment.heavy,
    USE.NAMES = FALSE,
    future.seed = TRUE
  )
  
  df$light <- future_mapply(
    replace_X,
    df$light,
    df$sequence_alignment.light,
    USE.NAMES = FALSE,
    future.seed = TRUE
  )
  
  df$X_heavy_after <- str_count(df$heavy, "X")
  df$X_light_after <- str_count(df$light, "X")
  
  summary <- df %>%
    summarise(
      total_X_heavy_before = sum(X_heavy_before),
      total_X_light_before = sum(X_light_before),
      total_X_heavy_after  = sum(X_heavy_after),
      total_X_light_after  = sum(X_light_after)
    )
  
  message(tag, " summary:")
  print(summary)
  
  df
}

positive_pairs <- process_pairs(positive_pairs, "positive")
negative_pairs <- process_pairs(negative_pairs, "negative")

message("Padding sequences...")
max_length_heavy <- max(nchar(positive_pairs$heavy))
max_length_light <- max(nchar(positive_pairs$light))

positive_pairs <- positive_pairs %>%
  mutate(
    heavy = str_pad(heavy, max_length_heavy, side = "right", pad = "!"),
    light = str_pad(light, max_length_light, side = "right", pad = "!")
  )

negative_pairs <- negative_pairs %>%
  mutate(
    heavy = str_pad(heavy, max_length_heavy, side = "right", pad = "!"),
    light = str_pad(light, max_length_light, side = "right", pad = "!")
  )

# adding labels
positive_pairs <- positive_pairs %>% mutate(label = 1)
negative_pairs <- negative_pairs %>% mutate(label = 0)

combined_pairs <- bind_rows(positive_pairs, negative_pairs) %>%
  mutate(heavy_light = paste0(heavy, "-", light)) %>%
  select(heavy_light, label)

out_file <- file.path(out_dir, "combined_test.tsv")
write_tsv(combined_pairs, out_file)

plan(sequential)

message("Done")
