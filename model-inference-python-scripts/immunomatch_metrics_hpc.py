#!/usr/bin/env python3
# Ishita Singh
# 3 March 2026

import os

n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))
os.environ["OMP_NUM_THREADS"] = str(n_cpus)
os.environ["MKL_NUM_THREADS"] = str(n_cpus)

print(f"Using {n_cpus} CPU cores")

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_auc_score, 
    roc_curve
)

TAG = "cdrl3"

file_path = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/pairing_scores_immunomatch_jaffe_test_cdrl3_shuffle.csv"
# file_path = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/pairing_scores_immunomatch_jaffe_test_vgene_igb.csv"

print("Reading data...")
df = pd.read_csv(file_path)

y_test = df["label"].values
y_pred_prob = df["pairing_scores"].values

# threshold at 0.5
y_pred = (y_pred_prob >= 0.5).astype("int8")

print("\nClassification Report\n")
print(classification_report(y_test, y_pred))

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Macro F1 Score:", f1_score(y_test, y_pred, average="macro"))

cm = confusion_matrix(y_test, y_pred)

# count and percentage in each square
total = cm.sum()
annot = [
    [f"{cm[i, j]}\n({cm[i, j] / total * 100:.1f}%)" for j in range(cm.shape[1])]
    for i in range(cm.shape[0])
]

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=annot,
    fmt="",  # in annot
    cmap="Blues",
    xticklabels=["Don't Pair", "Pair"],
    yticklabels=["Don't Pair", "Pair"]
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title(f"Immunomatch Confusion Matrix ({TAG})")
plt.savefig(f"immunomatch_{TAG}_confusion_matrix.png", dpi=300, bbox_inches="tight")
plt.close()

roc_score = roc_auc_score(y_test, y_pred_prob)
print("ROC AUC score:", roc_score)

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_score:.3f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Immunomatch ROC Curve ({TAG})")
plt.legend(loc="lower right")
plt.savefig(f"immunomatch_{TAG}_roc_curve.png", dpi=300, bbox_inches="tight")
plt.close()

print("Plots saved:")
print(f" - immunomatch_{TAG}_confusion_matrix.png")
print(f" - immunomatch_{TAG}_roc_curve.png")