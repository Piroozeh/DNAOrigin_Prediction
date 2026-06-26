
import pandas as pd

files = ["train.tsv", "dev.tsv", "test.tsv"]

pos_out = open("all_pos.fa", "w")
neg_out = open("all_neg.fa", "w")

counter = 0

for fname in files:
    df = pd.read_csv(fname, sep="\t")

    for i, row in df.iterrows():
        seq = row["sequence"].upper()
        label = row["label"]

        header = f">seq_{counter}"
        counter += 1

        if label == 1:
            pos_out.write(f"{header}\n{seq}\n")
        else:
            neg_out.write(f"{header}\n{seq}\n")

pos_out.close()
neg_out.close()

print("Created all_pos.fa and all_neg.fa")
