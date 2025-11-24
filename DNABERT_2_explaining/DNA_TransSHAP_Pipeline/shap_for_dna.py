import numpy as np
import torch
import logging

logging.basicConfig(format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S")
logging.getLogger().setLevel(logging.INFO)

class SHAPexplainerDNA:

     # self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def __init__(self, model, tokenizer,idx_to_token, token_to_idx):
        self.model = model
        self.tokenizer = tokenizer
        self.device ="cpu"
        # self.device =torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.model = model.to(self.device) 
        self.idx_to_token = idx_to_token      # index → token (e.g., "ATCG")
        self.token_to_idx = token_to_idx      # token → index

    def predict(self, indexed_inputs):
        """
        Accepts SHAP-style inputs: each row is a list of token indices.
        Masked positions (value 0) are replaced with a neutral token like [PAD].
        """
        # Convert indexes back to tokens, masking where needed
        reconstructed_tokens = [
            [self.idx_to_token[idx] if idx != 0 else '[PAD]' for idx in sequence]
            for sequence in indexed_inputs
        ]

        # Re-tokenize and convert to input IDs
        indexed_tokens = self.tokenize_and_index(reconstructed_tokens)
       
        tokens_tensor = torch.tensor(indexed_tokens).to(self.device)
        # print("shape of indexed_tokens:",(tokens_tensor.shape))
        with torch.no_grad():
            outputs = self.model(input_ids=tokens_tensor)
            predictions = outputs.logits.detach().cpu().numpy()

        final_probs = [self.softmax(x) for x in predictions]
      
        return np.array(final_probs)
    
    

    def softmax(self, logits):
        exps = np.exp(logits - np.max(logits))  # stable softmax
        return exps / np.sum(exps)

    def tokenize_and_index(self, list_of_token_lists, max_seq_len=None):
        """
        Takes a list of token strings (e.g., [["ATCG", "GATT", ...]]) and:
        - Converts to string
        - Tokenizes (BPE-aware)
        - Pads to max_seq_len
        """
        tokenized = [
            self.tokenizer.tokenize(" ".join(tokens))
            for tokens in list_of_token_lists
        ]

        if not max_seq_len:
            max_seq_len = min(max(len(t) for t in tokenized), 512)

        # Pad tokens
        padded_tokens = [
            t[:max_seq_len] + ['[PAD]'] * max(0, max_seq_len - len(t))
            for t in tokenized
        ]

        # Convert to input IDs
        indexed = [
            self.tokenizer.convert_tokens_to_ids(t)
            for t in padded_tokens
        ]

        return indexed

    def dt_to_idx(self, token_lists, max_seq_len=None):
        """
        Converts list of token strings to index arrays (for SHAP masking)
        """
        idx_sequences = [
            [self.token_to_idx[token] for token in tokens]
            for tokens in token_lists
        ]

        if not max_seq_len:
            max_seq_len = min(max(len(seq) for seq in idx_sequences), 512)

        padded = [
            seq + [0] * (max_seq_len - len(seq))
            if len(seq) < max_seq_len else seq[:max_seq_len]
            for seq in idx_sequences
        ]

        return np.array(padded), max_seq_len
