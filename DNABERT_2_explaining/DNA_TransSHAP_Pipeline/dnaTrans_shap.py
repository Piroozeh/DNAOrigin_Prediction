import shap
import torch
import pickle
import random
import os
import logging
import transformers
import pandas as pd
import numpy as np
from shap_for_dna import SHAPexplainerDNA

logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
logging.getLogger().setLevel(logging.INFO)

logging.getLogger("shap").setLevel(logging.WARNING)
shap.initjs()
# Step 1  Load model / test sequences
model_path = "/p/project1/hai_dnaori/piroozeh1/yeast-origins/data/OriDB_random_neg/train_dev_test_oridb/DNABERT2/model2"

config = transformers.BertConfig.from_pretrained(
                model_path,
                num_labels=2
        )
   
model = transformers.AutoModelForSequenceClassification.from_pretrained(model_path, config=config, from_tf=False,trust_remote_code=True)


tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_path,
        model_max_length=125,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True   
    )

#  Create a list of sequences to explain
texts=[]
data_dir_test="/p/project1/hai_dnaori/piroozeh1/yeast-origins/data/OriDB_random_neg/train_dev_test_oridb"
test = pd.read_csv(os.path.join(data_dir_test,"test_window.tsv"), sep='\t', header=0)
print(test.head())
texts.extend(test['seq'].tolist())
print("number of sequences: ", len(texts))

# Build vocab dictionaries
vocab_dict = tokenizer.get_vocab()
reverse_vocab_dict = {index: token for token, index in vocab_dict.items()}


# Step 2: Create SHAP-compatible predictor

predictor = SHAPexplainerDNA(
    model=model,
    tokenizer=tokenizer,
    idx_to_token=reverse_vocab_dict,
    token_to_idx=vocab_dict
)

# Step 3: Preprocess training data to token index arrays
data_dir_train="/p/project1/hai_dnaori/piroozeh1/yeast-origins/data/OriDB_random_neg/train_dev_test_oridb"
data = pd.read_csv(os.path.join(data_dir_train,"train_window.tsv"),sep='\t',header=0)

train_dt = [tokenizer.tokenize(seq) for seq in data['seq'].tolist()]  # Each seq becomes list of tokens
idx_train_data, max_seq_len = predictor.dt_to_idx(train_dt)
print("max seq len train: " , max_seq_len )

test_dt = [tokenizer.tokenize(seq) for seq in test['seq'].tolist()]  # Each seq becomes list of tokens
idx_test_data, max_seq_len_test = predictor.dt_to_idx(test_dt)
print("max seq len test: " , max_seq_len_test )

# Step 4: Initialize SHAP's KernelExplainer
explainer = shap.KernelExplainer(
    model=predictor.predict,
    data=shap.kmeans(idx_train_data, k=50)  # Representative background
)

# Step 5: Tokenize the DNA sequences to explain
# Preprocess target sequences to explain (same way)
texts_ = [tokenizer.tokenize(seq) for seq in texts]
idx_texts, _ = predictor.dt_to_idx(texts_, max_seq_len=max_seq_len)

nsamples=100
# l1_reg= "aic"
l1_reg= 0.01
print("Number of sequences to explain: ", len(idx_texts))
print("nsamples for SHAP: ", nsamples)
print("l1_reg for SHAP: ", l1_reg)
shap_values = explainer.shap_values(X=idx_texts, nsamples=nsamples, l1_reg=l1_reg)
print("shap values: ", shap_values)
# Step 6: Save SHAP values 


num_samples = len(texts_)
num_classes=2

flat_data = []
for sample_idx in range(num_samples):
    tokens = texts_[sample_idx]
    for class_idx in range(num_classes):
        shap_vals = shap_values[sample_idx, :len(tokens), class_idx]
        for token, shap_val in zip(tokens, shap_vals):
            flat_data.append({
                "sample_index": sample_idx,
                "class": class_idx,
                "token": token,
                "shap_value": shap_val
            })
# Convert to DataFrame
df_shap = pd.DataFrame(flat_data)

# Save to CSV
df_shap.to_csv("shap_values.csv", index=False)
print("Saved SHAP values to shap_values.csv")



# Save the shap values
save_path = "shap_values_all.pkl"
with open(save_path, "wb") as f:
    pickle.dump({
        "shap_values": shap_values,
        "expected_value": explainer.expected_value,
        "token_sequences": idx_texts,
        "tokens": texts_
    }, f)

print(f"Saved SHAP values for {len(idx_texts)} sequences to: {save_path}")
