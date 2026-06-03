#!/usr/bin/env python3
# Ishita Singh 
# 13th January, 2026
# HPC version dervied from ImmunoMatch Colab

import os
import argparse
import warnings
import shutil

import pandas as pd
import numpy as np
import torch

from functools import partial
from datasets import load_dataset
from transformers import (
    RoFormerTokenizer,
    RoFormerForSequenceClassification,
    Trainer,
    TrainingArguments,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

# for I/O
LAB_BASE = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab"

DATA_DIR = os.path.join(LAB_BASE, "data")
OUTPUT_DIR = os.path.join(LAB_BASE, "outputs")
TMP_DIR = os.path.join(LAB_BASE, "tmp")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_DIR, exist_ok=True)

# Functions for the calculation of the pairing score for single VH-VL
def preprocess_single_seq(seq):
  """
  Add the gap between adjacent amino acids, so that the model treat each amino acid as an individual token.
  args: input: seq: a string of amino acids
  """
  return " ".join(list(seq))


def pairing_score_single_pair(h_seq, l_seq, model_checkpoint):
    """
    Output the pairing score of a single pair of VH and VL sequences
    args:input:
    h_seq: a gapped sequence outputted from the function preprocess_seq
    l_seq: a gapped sequence outputted from the function preprocess_seq
    model_checkpoint: the checkpoint of the version of the immunoMatch of your interest
    """
    tokenizer = RoFormerTokenizer.from_pretrained(model_checkpoint)
    model = RoFormerForSequenceClassification.from_pretrained(model_checkpoint)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = tokenizer(
        h_seq,
        l_seq,
        return_tensors="pt",
        padding="max_length",
        max_length=256,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model(**inputs)

    pairing_score = torch.nn.functional.softmax(
        output.logits, dim=1
    )[0][1].item()

    return pairing_score


# Functions for the calculation of the pairing score for batches of VH-VL
def preprocess_seq(example, hseqcol="input_Hseq", lseqcol="input_Lseq"):
    return {
        "input_Hseq": " ".join(list(example[hseqcol])),
        "input_Lseq": " ".join(list(example[lseqcol])),
    }


def tokenize_function(
    examples,
    tokenizer,
    hseqcol="input_Hseq",
    lseqcol="input_Lseq",
    max_length=256,
    return_tensors="pt",
):
    return tokenizer(
        examples[hseqcol],
        examples[lseqcol],
        padding="max_length",
        truncation=True,
        max_length=max_length
        )


def tokenize_the_datasets(df_dir, hseq_col, lseq_col, tokenizer):
    """
    Tokenize the datasets
    args:input:
    df_dir: str, the directory of the dataset
    hseq_col: str, the column name of the heavy chain sequence
    lseq_col: str, the column name of the light chain sequence
    """
    df = pd.read_csv(df_dir)

    datasets = load_dataset("csv", data_files={"test": df_dir})
    datasets = datasets.map(
        partial(preprocess_seq, hseqcol=hseq_col, lseqcol=lseq_col)
    )
    datasets = datasets.map(
        partial(tokenize_function, tokenizer=tokenizer), batched=True
    )

    return df, datasets


def pairing_scores_batches(df_dir, hseq_col, lseq_col, model_checkpoint):
    """
    Load the model and make the pairing prediction on batches of sequences
    args:input:
    df_dir: the directory of the csv files holding the sequences of pairs of VH and VL sequences
    hseq_col: the column name of the column of VH sequences
    lseq_col: the column name of the column of VL sequences
    model_checkpoint: the chesck point of the version of ImmunoMatch of your interest
    """
    tokenizer = RoFormerTokenizer.from_pretrained(model_checkpoint)
    model = RoFormerForSequenceClassification.from_pretrained(model_checkpoint)

    df, tokenized_datasets = tokenize_the_datasets(
        df_dir, hseq_col, lseq_col, tokenizer
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    batch_size = 48

    args = TrainingArguments(
        output_dir=TMP_DIR,
        per_device_eval_batch_size=batch_size,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
    )

    pred_result = trainer.predict(tokenized_datasets["test"])

    pairing_scores = torch.nn.functional.softmax(
        torch.tensor(pred_result.predictions), dim=1
    )[:, 1].tolist()

    df["pairing_scores"] = pairing_scores
    return df

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--h_col", default="VH")
    parser.add_argument("--l_col", default="VL")
    parser.add_argument(
        "--ltype_col",
        default=None,
        help="Optional column for IGK / IGL splitting",
    )

    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)

    if args.ltype_col:
        kappa_data = data[data[args.ltype_col].str.contains("IGK")]
        lambda_data = data[data[args.ltype_col].str.contains("IGL")]

        kappa_path = os.path.join(TMP_DIR, "kappa_tmp.csv")
        lambda_path = os.path.join(TMP_DIR, "lambda_tmp.csv")

        kappa_data.to_csv(kappa_path, index=False)
        lambda_data.to_csv(lambda_path, index=False)

        k_res = pairing_scores_batches(
            kappa_path,
            args.h_col,
            args.l_col,
            "fraternalilab/immunomatch-kappa",
        )
        l_res = pairing_scores_batches(
            lambda_path,
            args.h_col,
            args.l_col,
            "fraternalilab/immunomatch-lambda",
        )

        result = pd.concat([k_res, l_res]).sort_index()

    else:
        result = pairing_scores_batches(
            args.input_csv,
            args.h_col,
            args.l_col,
            "fraternalilab/immunomatch",
        )

    result.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    main()
