import pandas as pd

def extract_best(infile, outfile):
    # Read file (skip comment lines if present)
    df = pd.read_csv(infile, sep="\t", comment="#")

    # Print columns so you can verify once
    print(f"Columns in {infile}:")
    print(df.columns.tolist())

    # Adjust these names if needed based on output
    seq_col = df.columns[0]
    score_col = df.columns[-2]
    pval_col = df.columns[-1]

    df[score_col] = pd.to_numeric(df[score_col], errors='coerce')
    df[pval_col] = pd.to_numeric(df[pval_col], errors='coerce')

    # Keep highest score per sequence
    idx = df.groupby(seq_col)[score_col].idxmax()
    best = df.loc[idx, [seq_col, score_col, pval_col]].copy()
    best.columns = ["sequence_id", "best_score", "best_pvalue"]

    best.to_csv(outfile, index=False)
    print(f"Saved {outfile}")

extract_best("acs_pos.scan.txt", "acs_pos.best.csv")
extract_best("acs_neg.scan.txt", "acs_neg.best.csv")