import pandas as pd
import numpy as np
import re

# -----------------------------
# Input files
# -----------------------------
test_file = "test.csv"
test_shuffled_file = "test_shuffled.csv"

tn_file = "TN_indices.csv"
tn_tn_file = "TN_remain_TN_indices.csv"
tn_fp_file = "TN_turnto_FP_indices.csv"
tp_file = "TP_indices.csv"

# -----------------------------
# SHAP-derived motifs from TP
# Replace or extend this list if needed
# -----------------------------
tp_motifs = [
    "TAAATTA", "TAGTTTT", "CTTAATT", "GTATA", "TGTG", "TTTTTTTT",
    "CACACACA", "GTGTG", "TATTAAA", "CTTTTA", "TATGTA", "TTATTA",
    "CTGTG", "GAGTG", "TGATTAAA", "TATG"
]

# -----------------------------
# Helper functions
# -----------------------------
def read_index_file(path):
    df = pd.read_csv(path)
    col = df.columns[0]
    return df[col].astype(int).tolist()

def at_fraction(seq):
    seq = seq.upper()
    return sum(1 for c in seq if c in {"A", "T"}) / len(seq)

def motif_count(seq, motifs):
    seq = seq.upper()
    return sum(seq.count(m.upper()) for m in motifs)

def longest_at_run(seq):
    seq = seq.upper()
    runs = re.findall(r"[AT]+", seq)
    return max((len(r) for r in runs), default=0)

def alternating_at_count(seq, min_len=4):
    """
    Counts occurrences of alternating A/T patterns, e.g. ATAT, TATA, ATATAT.
    Non-overlapping regex count of stretches length >= min_len.
    """
    seq = seq.upper()
    pattern = rf"(?=(([AT](?!\1))[AT]){{{max(1, min_len//2 - 1)},}})"
    # Simpler and more robust version:
    count = 0
    for i in range(len(seq) - min_len + 1):
        sub = seq[i:]
        m = re.match(r"(?:AT)+A?|(?:TA)+T?", sub)
        if m and len(m.group(0)) >= min_len:
            count += 1
    return count

def alternating_at_count_simple(seq, min_len=4):
    """
    Simpler count of windows starting at each position that begin with an
    alternating A/T tract of length >= min_len.
    """
    seq = seq.upper()
    count = 0
    n = len(seq)
    for i in range(n - min_len + 1):
        run_len = 1
        j = i + 1
        while j < n and seq[j] in {"A", "T"} and seq[j-1] in {"A", "T"} and seq[j] != seq[j-1]:
            run_len += 1
            j += 1
        if run_len >= min_len:
            count += 1
    return count

def compute_features(df, seq_col, motifs):
    out = df.copy()
    out["AT_fraction"] = out[seq_col].apply(at_fraction)
    out["motif_count"] = out[seq_col].apply(lambda s: motif_count(s, motifs))
    out["longest_AT_run"] = out[seq_col].apply(longest_at_run)
    out["alternating_AT_count"] = out[seq_col].apply(alternating_at_count_simple)
    return out

def zscore_columns(df, cols):
    out = df.copy()
    for c in cols:
        mu = out[c].mean()
        sd = out[c].std(ddof=0)
        if sd == 0:
            out[c + "_z"] = 0.0
        else:
            out[c + "_z"] = (out[c] - mu) / sd
    return out

# -----------------------------
# Load data
# -----------------------------
test_df = pd.read_csv(test_file)
test_shuf_df = pd.read_csv(test_shuffled_file)

# Add explicit index if not present
test_df = test_df.reset_index().rename(columns={"index": "orig_index"})
test_shuf_df = test_shuf_df.reset_index().rename(columns={"index": "orig_index"})

# -----------------------------
# Load groups
# -----------------------------
TN_idx = read_index_file(tn_file)
TN_TN_idx = read_index_file(tn_tn_file)
TN_FP_idx = read_index_file(tn_fp_file)
TP_idx = read_index_file(tp_file)

# -----------------------------
# Build grouped dataframes
# -----------------------------
TN_original = test_df[test_df["orig_index"].isin(TN_idx)].copy()
TP_original = test_df[test_df["orig_index"].isin(TP_idx)].copy()

TN_TN_shuffled = test_shuf_df[test_shuf_df["orig_index"].isin(TN_TN_idx)].copy()
TN_FP_shuffled = test_shuf_df[test_shuf_df["orig_index"].isin(TN_FP_idx)].copy()

TN_original["group"] = "TN_original"
TP_original["group"] = "TP_original"
TN_TN_shuffled["group"] = "TN_TN_shuffled"
TN_FP_shuffled["group"] = "TN_FP_shuffled"

# -----------------------------
# Compute features
# -----------------------------
TN_original = compute_features(TN_original, "sequence", tp_motifs)
TP_original = compute_features(TP_original, "sequence", tp_motifs)
TN_TN_shuffled = compute_features(TN_TN_shuffled, "sequence", tp_motifs)
TN_FP_shuffled = compute_features(TN_FP_shuffled, "sequence", tp_motifs)

all_df = pd.concat(
    [TN_original, TP_original, TN_TN_shuffled, TN_FP_shuffled],
    ignore_index=True
)

feature_cols = ["AT_fraction", "motif_count", "longest_AT_run", "alternating_AT_count"]
all_df = zscore_columns(all_df, feature_cols)

# Standardized unweighted index
z_cols = [c + "_z" for c in feature_cols]
all_df["AT_index"] = all_df[z_cols].mean(axis=1)

# -----------------------------
# Summaries
# -----------------------------
summary = all_df.groupby("group")[feature_cols + ["AT_index"]].agg(
    ["mean", "median", "std", "count"]
)

print("=== Group summary ===")
print(summary)

print("\n=== Mean AT_index by group ===")
print(all_df.groupby("group")["AT_index"].mean().sort_values(ascending=False))

print("\n=== Median AT_index by group ===")
print(all_df.groupby("group")["AT_index"].median().sort_values(ascending=False))

print("\n=== Label sanity checks ===")
print("Original TP labels:", TP_original["label"].value_counts().to_dict())
print("Original TN labels:", TN_original["label"].value_counts().to_dict())
print("Shuffled TN->TN labels:", TN_TN_shuffled["label"].value_counts().to_dict())
print("Shuffled TN->FP labels:", TN_FP_shuffled["label"].value_counts().to_dict())

# -----------------------------
# Optional: save per-sequence table
# -----------------------------
all_df.to_csv("AT_index_per_sequence.csv", index=False)
summary.to_csv("AT_index_group_summary.csv")

