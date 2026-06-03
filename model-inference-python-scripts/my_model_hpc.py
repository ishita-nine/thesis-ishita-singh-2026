#!/usr/bin/env python3
# Ishita Singh 
# 3rd March, 2026
# HPC version to run my model
# on 14th April, all models (not just RF were included)

import os

n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

os.environ["OMP_NUM_THREADS"] = str(n_cpus)
os.environ["MKL_NUM_THREADS"] = str(n_cpus)

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    f1_score
)

from sklearnex import patch_sklearn
patch_sklearn()

print(f"Using {n_cpus} CPU cores")

SEED = 9
TAG = "train_vgene_test_cdrl3"

# for vgene 
# train_path = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/combined_training_40K_plus_jaffe.tsv"
# test_path  = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/jaffe_combined_test_tab.tsv"

# for cdrl3
# train_path = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/jaffe_combined_training_cdrl3_shuffle.tsv"
# test_path  = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/jaffe_combined_test_cdrl3_shuffle.tsv"

# for vgene, combined training from 3 sources 
train_path = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/outputs/combined_training_pos_neg_3_sources.tsv"
# test_path  = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/jaffe_combined_test_tab.tsv"

# for cdrl3, combined training from 3 sources
# train_path = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/outputs/combined_training_pos_neg_3_sources_cdrl3_shuffle.tsv"
test_path  = "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/jaffe_combined_test_cdrl3_shuffle.tsv" # is a csv

# read_csv for csv's and read_table for tsv
df_train = pd.read_table(train_path, usecols=["heavy_light", "label"])
df_test  = pd.read_csv(test_path,  usecols=["heavy_light", "label"])

df_train["label"] = df_train["label"].astype("int8")
df_test["label"]  = df_test["label"].astype("int8")

amino_acids = list("ACDEFGHIKLMNPQRSTVWYX*!-.")

vectorizer = CountVectorizer(
    analyzer="char",
    vocabulary=amino_acids,
    lowercase=False
)

print("Vectorizing training data")
x_train = vectorizer.fit_transform(df_train["heavy_light"])
y_train = df_train["label"]

print("Vectorizing test data")
x_test = vectorizer.transform(df_test["heavy_light"])
y_test = df_test["label"]

def performance_metrics(model, model_name, x_test, y_test, dense=False):
    print(f"\nEvaluating {model_name}")

    if dense:
        x_test_input = x_test.toarray()
    else:
        x_test_input = x_test

    y_pred_prob = model.predict_proba(x_test_input)[:, 1]
    y_pred = (y_pred_prob >= 0.5).astype("int8")

    pd.DataFrame({
        "y_true": y_test.values,
        "y_pred_t50": y_pred,
        "y_pred_prob": y_pred_prob
    }).to_csv(f"{model_name}_{TAG}_predictions.tsv", sep="\t", index=False)
    print(f"saved predictions to {model_name}_{TAG}_predictions.tsv")

    print("\nClassification Report\n")
    print(classification_report(y_test, y_pred))

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Macro F1 Score:", f1_score(y_test, y_pred, average="macro"))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # showing count and percentage for each square
    total = cm.sum()
    annot = [
        [f"{cm[i, j]}\n({cm[i, j] / total * 100:.1f}%)" for j in range(cm.shape[1])]
        for i in range(cm.shape[0])
    ]

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="", # in annot
        cmap="Blues",
        xticklabels=["Don't Pair", "Pair"],
        yticklabels=["Don't Pair", "Pair"]
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model_name} Confusion Matrix (for {TAG})")
    plt.savefig(f"{model_name}_{TAG}_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()

    # ROC
    roc_score = roc_auc_score(y_test, y_pred_prob)
    print("ROC AUC score:", roc_score)

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {roc_score:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{model_name} ROC Curve (for {TAG})")
    plt.legend(loc="lower right")
    plt.savefig(f"{model_name}_{TAG}_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

print("Training Logistic Regression")
LR = LogisticRegression(max_iter=1000, random_state=SEED)
LR.fit(x_train, y_train)
performance_metrics(LR, "LR", x_test, y_test)

print("Training Random Forest")
RF = RandomForestClassifier(n_jobs=n_cpus, random_state=SEED)
RF.fit(x_train, y_train)
performance_metrics(RF, "RF", x_test, y_test)

print("Training Gaussian Naive Bayes")
NB = GaussianNB()
NB.fit(x_train.toarray(), y_train)
performance_metrics(NB, "NB", x_test, y_test, dense=True)

print("Training MLP")
MLP = MLPClassifier(random_state=SEED)
MLP.fit(x_train.toarray(), y_train)
performance_metrics(MLP, "MLP", x_test, y_test, dense=True)

# this was full SVM, never finished in time 
# print("Training RBF SVM")
# RBF_SVC = svm.SVC(kernel="rbf", probability=True, random_state=SEED)
# RBF_SVC.fit(x_train, y_train)
# performance_metrics(RBF_SVC, "SVM", x_test, y_test)

# svm doesn't scale to 1m rows, train on 100k subsample
SVM_SUBSAMPLE = 100_000
print(f"Training RBF SVM on stratified subsample of {SVM_SUBSAMPLE} rows")

# stratified subsample preserving 1:1 class balance
svm_train = (
    df_train.groupby("label", group_keys=False)
            .apply(lambda g: g.sample(n=SVM_SUBSAMPLE // 2, random_state=SEED))
)
print(f"SVM subsample label counts:\n{svm_train['label'].value_counts()}")

x_train_svm = vectorizer.transform(svm_train["heavy_light"])
y_train_svm = svm_train["label"]

RBF_SVC = svm.SVC(kernel="rbf", probability=True, random_state=SEED)
RBF_SVC.fit(x_train_svm, y_train_svm)
performance_metrics(RBF_SVC, "SVM", x_test, y_test)