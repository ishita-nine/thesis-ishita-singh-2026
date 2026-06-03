#!/usr/bin/env Rscript

# Author: Ishita Singh 
# Date: 2/28/26
# Term: 26W
# Project: Senior Thesis
# Purpose: Prep positive pairs from a igblast output airr table with translated sequence column 
# Note that the output is not processed (say X replacement for either my model or immunomatch)
# Location: To be run from thesis-venv in Discovery scratch, data in lab directory 
# Can also be run locally for small datasets just setwd to data directory

# On 4/26 I reran this script on data/test_HL_data_translated.tsv and data/translated_HL_data_translated.tsv as a sanity check 
# for test I used new_barcode (snake case not camel case as the unique id)
# for training I used sample + cell id as sample cell id 

library(dplyr)
library(alakazam)
library(airr)
library(readr)
library(stringr)

# if running locally: session -> setwd -> choose directory and choose the specific folder
# data_dir <- "/Volumes/rc/lab/H/HoehnK/Ishita/immunomatch-lab/data"
# setwd(data_dir)

# data_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe"
# data <- read_rearrangement(file.path(data_dir, "igblast/jaffe_test_sample_cell_id_translated.tsv"))

data_dir <- "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe"
data <- read_rearrangement(file.path(data_dir, "igblast/jaffe_training_sample_cell_id_translated.tsv"))

# need a unique cell ID
# data <- data %>% rename(sample_cell_id = new_barcode)
# data <- data %>%
#   mutate(sample_cell_id = paste0(sample, "_", cell_id))

# remove cells cells with multiple heavy chains
multi_heavy <- table(dplyr::filter(data, locus == "IGH")$sample_cell_id)
multi_heavy_cells <- names(multi_heavy)[multi_heavy > 1]
data <- dplyr::filter(data,!sample_cell_id %in% multi_heavy_cells)

# separate heavy and light chains
h <- data %>% filter(locus == "IGH")
l <- data %>% filter(locus != "IGH")

# sanity check - should print TRUE
print(all(table(h$sample_cell_id) == 1))

# extract v genes
h <- h %>% mutate(v_gene = sapply(strsplit(v_call, ","), alakazam::getGene))
l <- l %>% mutate(v_gene = sapply(strsplit(v_call, ","), alakazam::getGene))

# resolve ambiguous V gene annotations to simplest/shortest gene name
simplify_v_gene <- function(x) {
  x <- unlist(x)
  x[nchar(x) == min(nchar(x))][1]
}
h <- h %>% mutate(v_gene = sapply(v_gene, simplify_v_gene))
l <- l %>% mutate(v_gene = sapply(v_gene, simplify_v_gene))

# keep positive pairs
positive_pairs <- inner_join(h, l, by = "sample_cell_id", suffix = c(".heavy", ".light")) %>%
  rename(
    vgh = v_gene.heavy,
    vgl = v_gene.light,
    heavy = translated_sequence.heavy,
    light = translated_sequence.light
  )

write_tsv(positive_pairs, file.path(data_dir, "outputs/jaffe_training_positive_sanity_check.tsv"))
