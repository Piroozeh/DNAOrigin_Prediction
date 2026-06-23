import os
import csv
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
import random
import torch
import transformers

import numpy as np
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from tqdm import tqdm 
# from transformers import AutoConfig
# from transformers import AutoModel
# from peft import (
#     LoraConfig,
#     get_peft_model,
#     get_peft_model_state_dict,
# )


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="finetuned_DNABERT_2")



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})

    atte_rep_mode: int = field(default=0, metadata={"help": "Mode for attention score representation: 0 or 1 or 2. 0:extract attention scores for token CLS and sum over all heads. 1:  sum all heads and average over query tokens  2: sum over all heads, output is 2 dim matrix: tokenized_length * tokenized_length"})
    test_file: str = field(default=0, metadata={"help": "Whether to use test file for attention extraction."})
   
@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
  
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
  
    fp16: bool = field(default=False)
 
    output_dir: str = field(default="output")
 
    seed: int = field(default=42)



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, 
                 data_path: str, 
                 tokenizer: transformers.PreTrainedTokenizer, 
                 kmer: int = -1):

        super(SupervisedDataset, self).__init__()

        # load data from the disk
        with open(data_path, "r") as f:
            data = list(csv.reader(f))[1:]
        if len(data[0]) == 2:
            # data is in the format of [text, label]
            logging.warning("Perform single sequence classification...")
            texts = [d[0] for d in data]
            labels = [int(d[1]) for d in data]
        elif len(data[0]) == 3:
            # data is in the format of [text1, text2, label]
            logging.warning("Perform sequence-pair classification...")
            texts = [[d[0], d[1]] for d in data]
            labels = [int(d[2]) for d in data]
        else:
            raise ValueError("Data format not supported.")
        

        output = tokenizer(
            texts,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
            return_attention_mask=True 
        )
        
        self.input_ids = output["input_ids"]
        self.attention_mask = output["attention_mask"]
        self.labels = labels
        self.num_labels = len(set(labels))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        
        return dict(input_ids=self.input_ids[i], attention_mask=self.attention_mask[i], labels = torch.tensor(self.labels[i]) if not isinstance(self.labels[i], torch.Tensor) else self.labels[i])




def calcute_attention():
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("data path", data_args.data_path)

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if training_args.n_gpu > 0:
        torch.cuda.manual_seed_all(training_args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    # Load model directly

    config = transformers.BertConfig.from_pretrained(
                model_args.model_name_or_path,
                num_labels=2,
                cache_dir=training_args.cache_dir if training_args.cache_dir else None
            )
    config.output_attentions = True
  

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=False,
        config=config,
        trust_remote_code=True,
        cache_dir=training_args.cache_dir if training_args.cache_dir else None
    )
    model.to(device) 

     # choose target data
    if data_args.test_file:
        print("Using test file for attention extraction")
        train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "test.csv"), 
                                      kmer=data_args.kmer)
    else:
        print("Using train file for attention extraction")
        train_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                      data_path=os.path.join(data_args.data_path, "train.csv"), 
                                      kmer=data_args.kmer)
    
    

    decoded_texts = [tokenizer.decode(input_ids, skip_special_tokens=False) for input_ids in train_dataset.input_ids]
    with open(os.path.join(training_args.output_dir, "tokenized_test_set.txt"), "w") as f:
        for text in decoded_texts:
            f.write(text + "\n")
    # print(decoded_texts[0])
    # print(len(decoded_texts[0].split()))
     
    tokennized_seq_len=len(decoded_texts[0].split())
    seq_len=4 * int(training_args.model_max_length)
    # print(seq_len)
    preds = np.zeros([len(train_dataset),2])

# Create a DataLoader for the  dataset
    
    batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)
    train_loader = DataLoader(train_dataset, batch_size)
    

    preds = np.zeros([len(train_dataset),2])
    attention_scores = np.zeros([len(train_dataset), 12, tokennized_seq_len, tokennized_seq_len])
    softmax = torch.nn.Softmax(dim=1)
# Run the model on the dataset to extract attention scores and predictions
   
    if training_args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)
    for index, batch in enumerate(tqdm(train_loader, desc="Processing batches")):
        model.eval()
        # batch = tuple(t.to(device) for t in batch)
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
    
        with torch.no_grad():
            input_ids = batch['input_ids'].to(training_args.device)
            attention_mask = batch['attention_mask'].to(training_args.device)
            labels = batch['labels'].to(training_args.device)
            # attention_mask = batch['attention_mask'].to(training_args.device)

           
            outputs = model(input_ids, attention_mask=attention_mask, output_attentions=True, return_dict=True)
            
            attentions = outputs.attentions
            last_layer_attention = attentions[-1]  
            attention_scores[index*batch_size:index*batch_size+len(batch['input_ids']),:,:,:] = last_layer_attention.cpu().numpy()
            

            logits = outputs.logits
            preds[index*batch_size:index*batch_size+len(batch['input_ids']),:] = logits.detach().cpu().numpy()
            
            # probabilities = torch.sigmoid(logits)
            # preds[index*batch_size:index*batch_size+len(batch['input_ids']),:] = probabilities.detach().cpu().numpy()
    probs = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()   
    # probabilities = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
    np.save(os.path.join(training_args.output_dir, "pred_results.npy"), probs)
    print("Probs shape:")
    print(probs.shape)
   

    if data_args.atte_rep_mode == 0:
        print("Extract attention scores for token CLS and sum over all heads")
        
        CLS_attention_scores=np.zeros([len(train_dataset), tokennized_seq_len])
        for index, attention_score in enumerate(attention_scores):
            attn_score = []
            for i in range(0, attention_score.shape[-1]):
                attn_score.append(float(attention_score[:,0,i].sum())) # sum over all heads for token CLS
            
            sum_attention_scores = np.array(attn_score)
            CLS_attention_scores[index,:] = sum_attention_scores
        repr_attention_scores=CLS_attention_scores
        np.save(os.path.join(training_args.output_dir, "atten_token_based_cls.npy"), repr_attention_scores)
        print("CLS_attention_Score shape: ", repr_attention_scores.shape)

        # Convert token-level attention scores to base-level attention scores
        special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
        per_Base_attentions=np.zeros([len(train_dataset), seq_len])
        for index, input_ids in enumerate(train_dataset.input_ids):
            bp_attention = []
            decoded_text = tokenizer.decode(train_dataset.input_ids[index], skip_special_tokens=False)
            tokens = decoded_text.split()
            scores= repr_attention_scores[index]

            for token, score in zip(tokens, scores):
                if token not in special_tokens:
                # For each base pair in the token, assign the same attention score
                    for bp in token:
                        bp_attention.append(score)
            per_Base_attentions[index]=bp_attention

        print(per_Base_attentions[0]) 
        np.save(os.path.join(training_args.output_dir, "atten_per_base_cls.npy"), per_Base_attentions) 

    elif  data_args.atte_rep_mode == 1:
        print("Sum over heads and average over query tokens")
        # Sum over heads
        summed_heads = np.sum(attention_scores, axis=1) # 
        # Average over query tokens
        avg_attention_scores = np.mean(summed_heads, axis=(1)) 
        repr_attention_scores=avg_attention_scores
        np.save(os.path.join(training_args.output_dir, "atten_token_based_avrg_alltokens.npy"), repr_attention_scores)
        print("avrg_attention_Score shape: ", repr_attention_scores.shape)   

        # Convert token-level attention scores to base-level attention scores
        special_tokens = ["[CLS]", "[SEP]", "[PAD]"]
        per_Base_attentions=np.zeros([len(train_dataset), seq_len])
        for index, input_ids in enumerate(train_dataset.input_ids):
            bp_attention = []
            decoded_text = tokenizer.decode(train_dataset.input_ids[index], skip_special_tokens=False)
            tokens = decoded_text.split()
            scores= repr_attention_scores[index]

            for token, score in zip(tokens, scores):
                if token not in special_tokens:
                # For each base pair in the token, assign the same attention score
                    for bp in token:
                        bp_attention.append(score)
            per_Base_attentions[index]=bp_attention

        print(per_Base_attentions[0]) 
        np.save(os.path.join(training_args.output_dir, "atten_per_base_avrg_alltokens.npy"), per_Base_attentions) 
   

    elif data_args.atte_rep_mode == 2:
        print("Sum over all heads, output is 2 dim matrix: tokenized_length * tokenized_length")
        summed_heads = np.sum(attention_scores, axis=1) # sum over all heads
        repr_attention_scores=summed_heads
        np.save(os.path.join(training_args.output_dir, "atten_token_based_matrix.npy"), repr_attention_scores)
        print("head_sum_attention_Score shape:", summed_heads.shape)  

   
    
   
   
    




if __name__ == "__main__":
    calcute_attention()
