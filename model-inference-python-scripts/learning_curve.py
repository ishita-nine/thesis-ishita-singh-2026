#!/usr/bin/env python3
# Ishita Singh 
# 7th May, 2026
# learning curve: auc vs training set size, accuracy vs training set size 
# random forest only, two curves on each plot (v gene and cdrl3)
# uses combined-3-sources training data for both methods

import os

n_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))

os.environ["OMP_NUM_THREADS"] = str(n_cpus)
os.environ["MKL_NUM_THREADS"] = str(n_cpus)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib import font_manager

# register bundled cmu serif font (works locally and on hpc)
script_dir = os.path.dirname(os.path.abspath(__file__))
font_dir = os.path.join(script_dir, "fonts")
if os.path.isdir(font_dir):
    for fp in os.listdir(font_dir):
        if fp.lower().endswith((".ttf", ".otf")):
            font_manager.fontManager.addfont(os.path.join(font_dir, fp))

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["CMU Serif", "Computer Modern Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "cm"

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

from sklearnex import patch_sklearn
patch_sklearn()

print(f"using {n_cpus} cpu cores")

SEED = 9
N_POINTS = 12 # number of training subsets used  
THRESHOLDS = [0.5, 0.65, 0.8] # to compare accuracy thresholds 

# paths: combined-3-sources training, jaffe test set for each method
DATASETS = {
    "V gene": {
        "train": "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/outputs/combined_training_pos_neg_3_sources.tsv",
        "test":  "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/jaffe_combined_test_tab.tsv",
        "test_is_csv": False,  # actual tsv
    },
    "Random shuffle": {
        "train": "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/outputs/combined_training_pos_neg_3_sources_cdrl3_shuffle.tsv",
        "test":  "/dartfs/rc/lab/H/HoehnK/Ishita/immunomatch-lab/jaffe/outputs/jaffe_combined_test_cdrl3_shuffle.tsv",
        "test_is_csv": True,  # is a CSV sorry 
    },
}

amino_acids = list("ACDEFGHIKLMNPQRSTVWYX*!-.")

vectorizer = CountVectorizer(
    analyzer="char",
    vocabulary=amino_acids,
    lowercase=False,
)

def load_data(method, file_info):
    print(f"\n loading {method} data")
    df_train = pd.read_table(file_info["train"], usecols=["heavy_light", "label"])
    reader = pd.read_csv if file_info["test_is_csv"] else pd.read_table
    df_test = reader(file_info["test"], usecols=["heavy_light", "label"])
    df_train["label"] = df_train["label"].astype("int8")
    df_test["label"] = df_test["label"].astype("int8")
    print(f"{method}: {len(df_train)} train rows, {len(df_test)} test rows")
    return df_train, df_test


# load both datasets up front so the shared training-size cap is computed from real data
loaded = {method: load_data(method, file_info) for method, file_info in DATASETS.items()}

# i ran it with the max size logic NOT commented out to get the full dataset, then explicitly capped at 400k, 100k, and 20k
# i then ran it on the full dataset again, but plotted in log space 
# cap = smaller of the two training set sizes, rounded down to even for exact class split
max_size = min(len(loaded[m][0]) for m in loaded)
max_size = (max_size // 2) * 2
# max_size = 20_000

# 10 evenly spaced sizes from max/10 up to max — identical x-axis for both curves
# linspace 
# train_sizes = np.linspace(max_size / N_POINTS, max_size, N_POINTS).astype(int)
# logspace 
train_sizes = np.logspace(np.log10(200), np.log10(max_size), N_POINTS).astype(int)
train_sizes = (train_sizes // 2) * 2  # all even for clean stratified split
print(f"\nshared max training size: {max_size}")
print(f"training sizes ({N_POINTS} points): {train_sizes.tolist()}")

results = []

# for each method (cdrl3 vs vgene)
for method, (df_train, df_test) in loaded.items():

    print(f"vectorizing {method} full training set (will slice for each size)")
    x_train_full = vectorizer.fit_transform(df_train["heavy_light"])
    y_train_full = df_train["label"].values
 
    print(f"vectorizing {method} test set")
    x_test = vectorizer.transform(df_test["heavy_light"])
    y_test = df_test["label"].values

    # pre-shuffle indices per class once, then take the first 'half' for each size.
    # this keeps subsamples nested (smaller is a subset of larger) and reproducible.
    pos_idx = np.where(y_train_full == 1)[0]
    neg_idx = np.where(y_train_full == 0)[0]
    rng = np.random.default_rng(SEED)
    pos_shuf = rng.permutation(pos_idx)
    neg_shuf = rng.permutation(neg_idx)
    print(f"{method} class counts: pos={len(pos_idx)}, neg={len(neg_idx)}")

    # for each subset within a method
    for size in train_sizes:
        half = int(size) // 2
        # safety: make sure each class has enough rows
        if half > len(pos_shuf) or half > len(neg_shuf):
            print(f"warning: requested half={half} exceeds class size, skipping")
            continue
        sel = np.concatenate([pos_shuf[:half], neg_shuf[:half]])
        x_tr = x_train_full[sel]
        y_tr = y_train_full[sel]
        actual_size = x_tr.shape[0]

        print(f"\ntraining RF on {method} with {actual_size} rows")
        rf = RandomForestClassifier(n_jobs=n_cpus, random_state=SEED)
        rf.fit(x_tr, y_tr)

        # for normal learning curve 
        y_pred_prob = rf.predict_proba(x_test)[:, 1]
        y_pred = (y_pred_prob >= 0.5).astype("int8")
        auc = roc_auc_score(y_test, y_pred_prob)
        acc = accuracy_score(y_test, y_pred)
        print(f"{method} size={actual_size} auc={auc:.4f} acc={acc:.4f}")
        results.append({"method": method, "size": actual_size, "auc": auc, "acc": acc})

        # for diff accuracy thresholds learning curve
        # y_pred_prob = rf.predict_proba(x_test)[:, 1]
        # auc = roc_auc_score(y_test, y_pred_prob)
        # row = {"method": method, "size": actual_size, "auc": auc}
        # for t in THRESHOLDS:
        #     row[f"acc_t{t}"] = accuracy_score(y_test, (y_pred_prob >= t).astype("int8"))
        # acc_str = " ".join([f"acc@{t}={row[f'acc_t{t}']:.4f}" for t in THRESHOLDS])
        # print(f"{method} size={actual_size} auc={auc:.4f} {acc_str}")
        # results.append(row)

    # free per-method memory before moving on
    del x_train_full, y_train_full, x_test, y_test, df_train, df_test

# save numeric results incase plotting needs to be redone 
results_df = pd.DataFrame(results)
# results_df.to_csv("learning_curve_results_3_acc_vgene.tsv", sep="\t", index=False)
# print("\nsaved results to learning_curve_results_3_acc_vgene.tsv")
print(results_df)

# one figure per metric, both methods on the same axes
# for metric, ylabel, fname in [
#     ("auc", "ROC AUC", "learning_curve_auc_log.png"),
#     ("acc", "Accuracy (threshold = 0.5)", "learning_curve_acc_log.png"),]:
#     plt.figure(figsize=(7, 5))
#     for method in DATASETS.keys():
#         sub = results_df[results_df["method"] == method].sort_values("size")
#         plt.plot(sub["size"], sub[metric], marker="o", linewidth=2, label=method)
#     plt.xscale("log")
#     plt.xlabel("training set size")
#     plt.ylabel(ylabel)
#     plt.title(f"Random Forest learning curve — {ylabel}")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#     plt.tight_layout()
#     plt.savefig(fname, dpi=300, bbox_inches="tight")
#     plt.close()
#     print(f"saved plot to {fname}")

# only auc 
plt.figure(figsize=(7, 5))
for method in DATASETS.keys():
    sub = results_df[results_df["method"] == method].sort_values("size")
    plt.plot(sub["size"], sub["auc"], marker="o", linewidth=2, label=method)
plt.xscale("log")
plt.xlabel("Training set size")
plt.ylabel("AUC-ROC on test set")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("learning_curve_auc_log.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved plot to learning_curve_auc_log.png")

# v gene only: 3 acc thresholds vs training size
# plt.figure(figsize=(8, 6))
# colors = {0.5: "C0", 0.65: "C1", 0.8: "C2"}
# sub = results_df[results_df["method"] == "V GENE"].sort_values("size")
# for t in THRESHOLDS:
#     plt.plot(sub["size"], sub[f"acc_t{t}"],
#              marker="o", linewidth=2,
#              color=colors[t],
#              label=f"threshold = {t}")
# plt.xscale("log")
# plt.xlabel("training set size")
# plt.ylabel("Accuracy")
# plt.title("Random Forest accuracy at different thresholds (V GENE)")
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig("learning_curve_3_acc_vgene.png", dpi=300, bbox_inches="tight")
# plt.close()
# print("saved plot to learning_curve_3_acc_vgene.png")