# Biologically informed negative sampling for antibody chain pairing classification
**Ishita Singh&ensp;|&ensp;
Computer Science&ensp;|&ensp;
Honors Thesis&ensp;|&ensp;
Co-advisors Soroush Vosoughi and Kenneth Hoehn** 

<img width="960" height="360" alt="image" src="https://github.com/user-attachments/assets/0a0aa931-736a-459b-9bae-58c1869af3da" />

<sub><sup>*Image Credits: Charlotte Gurr, The Pharmaceutical Journal*</sup></sub>

## Source data
Three public datasets of single-cell human B cell repertoires were used to train and test models in this thesis. The data from Jaffe et al. (2022) was downloaded from Figshare (doi:10.25452/figshare.plus.20338177). The data from Hoehn et al. (2021) and Lopes de Assis et al. (2023) are available on NCBI's Gene Expression Omnibus under accession numbers GSE164381 and GSE219098, respectively. T

## Code, computational environment, and intermediate data availability
For reproducibility, the code used to process data, train classical models, run experiments, and generate figures is available in this repository under a CC BY 4.0 license.

Due to size limits, intermediate files with the positive pairs from Hoehn et al. (2021) and Lopes de Assis et al. (2023) are made available in `intermediate-files`. Data processed from Jaffe et al. (2022) as well as any other materials of interest are available upon request. 

All scripts were written in R (v4.5.3) or Python (v3.9.25) and executed as Slurm jobs on Dartmouth's HPC cluster ‘Discovery’. Depending on the task, 4–16 CPU cores were used, and data was processed parallely where possible.

## Thesis abstract 
Antibody heavy and light chain (H/L) pairing is fundamental to antigen recognition and stability. While single-cell sequencing preserves native pairing information, widely used bulk repertoire and spatial transcriptomics platforms do not, motivating the need for efficient machine learning (ML) methods to infer H/L pairing. Training a binary classifier for this task faces the methodological challenge of a lack of true biological negatives, since natural selection eliminates B cells with incompatible H/L pairs.

In this thesis, I introduce a biologically informed negative sampling strategy for H/L pairing classification, drawing on known V-gene biases in heavy and light chain pairing. Pseudo-negatives are constructed by sampling H/L sequences whose V-gene combinations are absent from the observed data. I train five classical ML models on amino acid composition features of H/L variable region sequences, and benchmark them against a state-of-the-art language model and a commonly used random shuffling strategy.

The results demonstrate that chain pairing can be predicted using relatively simple and interpretable classical approaches, with performance strongly dependent on biologically informed V-gene-based pseudo-negative sampling.

I contribute to two parallel efforts in computational immunology. First, the growing body of ML methods to pair, and generate paired, antibody data. Second, demonstrating the importance of negative sampling in a field where poor feature engineering, in the absence of true negatives, has led to reduced generalizability and overestimations of model performance.
