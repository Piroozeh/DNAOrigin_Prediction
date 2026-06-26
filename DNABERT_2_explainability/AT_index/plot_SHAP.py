import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import mannwhitneyu
import pandas as pd
import numpy as np

all_df=pd.read_csv("AT_index_per_sequence.csv")
# Order groups logically
order = ["TP_original", "TN_FP_shuffled", "TN_original", "TN_TN_shuffled"]

TP = all_df[all_df["group"] == "TP_original"]["AT_index"]
TN = all_df[all_df["group"] == "TN_original"]["AT_index"]
TN_FP = all_df[all_df["group"] == "TN_FP_shuffled"]["AT_index"]
TN_TN = all_df[all_df["group"] == "TN_TN_shuffled"]["AT_index"]

print("\n=== Sample sizes ===")
print(f"TP_original: {len(TP)}")
print(f"TN_original: {len(TN)}")
print(f"TN_FP_shuffled: {len(TN_FP)}")
print(f"TN_TN_shuffled: {len(TN_TN)}")

#plt.figure(figsize=(7,5))

#sns.boxplot(
#    data=all_df,
#    x="group",
#    y="AT_index",
#    order=order,
#    palette=["#d62728","#ff7f0e","#1f77b4","#2ca02c"]
#)

#sns.stripplot(
#    data=all_df,
#    x="group",
#    y="AT_index",
#    order=order,
#    color="black",
#    alpha=0.5,
#    jitter=True
#)

#plt.ylabel("AT-richness index")
#plt.xlabel("")
#plt.title("AT-rich sequence signal across sequence groups")

#plt.xticks(rotation=20)

#plt.tight_layout()
#plt.savefig("AT_index_groups.png", dpi=300)
#plt.show()

#print("\n=== Sample sizes ===")
#print(f"TP_original: {len(TP)}")
#print(f"TN_original: {len(TN)}")
#print(f"TN_FP_shuffled: {len(TN_FP)}")
#print(f"TN_TN_shuffled: {len(TN_TN)}")

print("\n=== Statistical tests (Mann–Whitney U) ===")

def test(a, b, name_a, name_b):
    stat, p = mannwhitneyu(a, b, alternative="two-sided")
    print(f"{name_a} vs {name_b}: p = {p:.4e}")

TP = all_df[all_df.group=="TP_original"]["AT_index"]
TN = all_df[all_df.group=="TN_original"]["AT_index"]
TN_FP = all_df[all_df.group=="TN_FP_shuffled"]["AT_index"]
TN_TN = all_df[all_df.group=="TN_TN_shuffled"]["AT_index"]

test(TP, TN, "TP", "TN")
test(TP, TN_FP, "TP", "TN->FP")
test(TN, TN_FP, "TN", "TN->FP")
test(TN, TN_TN, "TN", "TN->TN")

def cliffs_delta(x, y):
    """
    Compute Cliff's delta effect size.
    """
    x = np.array(x)
    y = np.array(y)

    n_x = len(x)
    n_y = len(y)

    greater = 0
    smaller = 0

    for xi in x:
        greater += np.sum(xi > y)
        smaller += np.sum(xi < y)

    delta = (greater - smaller) / (n_x * n_y)
    return delta


def interpret_delta(delta):
    d = abs(delta)
    if d < 0.147:
        return "negligible"
    elif d < 0.33:
        return "small"
    elif d < 0.474:
        return "medium"
    else:
        return "large"


print("\n=== Effect sizes (Cliff's delta) ===")

def effect(a, b, name_a, name_b):
    delta = cliffs_delta(a, b)
    print(f"{name_a} vs {name_b}: delta = {delta:.3f} ({interpret_delta(delta)})")

effect(TP, TN, "TP", "TN")
effect(TP, TN_FP, "TP", "TN->FP")
effect(TN, TN_FP, "TN", "TN->FP")
effect(TN, TN_TN, "TN", "TN->TN")
