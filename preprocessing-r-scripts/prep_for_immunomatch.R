#!/usr/bin/env Rscript

# Author: Ishita Singh 
# Date: 1/27/25
# Term: 26W
# Project: Senior Thesis
# Purpose: Prep first 10K sequences from test set (total 260K) for immunomatch
# Location: To be run from R-home in Discovery scratch, data in lab directory - outputs
# CANNOT be locally due to large file sizes of this dataset 

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(stringr)
})

out_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs"

pos_file <- file.path(out_dir, "jaffe_test_positive_cdrl3_shuffle.tsv")
neg_file <- file.path(out_dir, "jaffe_test_negative_cdrl3_shuffle.tsv")

out_file <- file.path(out_dir, "immunomatch_jaffe_test_combined_cdrl3_shuffle.csv")

# reading the positive and negative pairs, and adding label column
pos <- read_tsv(
  pos_file,
  show_col_types = FALSE,
  col_select = c(heavy, light, locus.light)
) %>% mutate(label = 1)

# NOTE changed from read_tsv to read_csv because jaffe_test_neg_vgene file is a csv even though it is named .tsv
# changed back to read_tsv when working with cdrl3 based neg pair data 
neg <- read_tsv(
  neg_file,
  show_col_types = FALSE,
  col_select = c(heavy, light, locus.light)
) %>% mutate(label = 0)

df <- bind_rows(pos, neg)

df <- df %>%
  transmute(
    VH = gsub("X", "", heavy, fixed = TRUE),
    VL = gsub("X", "", light, fixed = TRUE),
    light_type = case_when(
      locus.light == "IGK" ~ "IGK",
      locus.light == "IGL" ~ "IGL",
      TRUE ~ NA_character_
    ),
    label = label
  )

# checking that there is a heavy, light, light_type, and allowed label
stopifnot(!any(is.na(df$VH)))
stopifnot(!any(is.na(df$VL)))
stopifnot(!any(is.na(df$light_type)))
stopifnot(all(df$label %in% c(0, 1)))

write_csv(
  df,
  out_file,
  quote = "all"
)
