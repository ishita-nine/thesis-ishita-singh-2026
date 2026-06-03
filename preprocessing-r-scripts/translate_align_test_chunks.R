#!/usr/bin/env Rscript

# Author: Ishita Singh 
# Date: 1/18/25
# Term: 26W
# Project: Senior Thesis
# Purpose: Translate sequence alignments for TEST data (~260K sequences) using alakazam translateDNA with chunks
# On 2/15 using this script to run on Jaffe training and test data 
# On 2/28 this script was used to run on Jaffee training and test data again, this time with a sample_cell id column
# Location: To be run from thesis-venv in Discovery scratch, data in lab directory 

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(alakazam)
  library(future)
  library(future.apply)
})

# set up 7-core parallel plan, will ask for 8 cores on slurm
plan(multisession, workers = 7)

# data_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/igblast"
# infile   <- file.path(data_dir, "jaffe_training_changeo_data_sample_cell_id.tsv")
# outfile  <- file.path(data_dir, "jaffe_training_sample_cell_id_translated.tsv")

# change made for ease of running on both test/traing data, read file names from command-line arguments
args <- commandArgs(trailingOnly = TRUE)
if(length(args) < 2){
  stop("Please provide infile and outfile as arguments: Rscript translate_align_test_chunks.R infile.tsv outfile.tsv")
}

infile  <- args[1]
outfile <- args[2]

chunk_size <- 20000

header <- read_lines(infile, n_max = 1)
col_names <- strsplit(header, "\t")[[1]]

write_lines(paste0(header, "\ttranslated_sequence"), outfile)

skip_rows <- 1
repeat {
  df <- read_tsv(
    infile,
    skip = skip_rows,
    n_max = chunk_size,
    col_names = col_names,
    show_col_types = FALSE
  )
  if (nrow(df) == 0) break
  
  df$translated_sequence <- unlist(
    future_lapply(
      df$sequence_alignment,
      function(x) if (is.na(x) || x == "") NA_character_ else translateDNA(x)
    ),
    use.names = FALSE
  )
  
  write_tsv(df, outfile, append = TRUE, col_names = FALSE)
  skip_rows <- skip_rows + nrow(df)
}

# clean up future workers
plan(sequential)


