import pandas as pd
from scipy import stats

pos = pd.read_csv("acs_pos.best.csv")
neg = pd.read_csv("acs_neg.best.csv")

print("=== BEST SCORE COMPARISON ===")
u_score = stats.mannwhitneyu(pos["best_score"], neg["best_score"])
print("Mann-Whitney p-value:", u_score.pvalue)

ks_score = stats.ks_2samp(pos["best_score"], neg["best_score"])
print("KS test p-value:", ks_score.pvalue)

print("\n=== BEST P-VALUE COMPARISON ===")
u_p = stats.mannwhitneyu(pos["best_pvalue"], neg["best_pvalue"])
print("Mann-Whitney p-value:", u_p.pvalue)

ks_p = stats.ks_2samp(pos["best_pvalue"], neg["best_pvalue"])
print("KS test p-value:", ks_p.pvalue)
