import os
import csv
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Tuple, List
import random
import torch
import transformers
from transformers.models.bert.configuration_bert import BertConfig
import sklearn
from sklearn.preprocessing import normalize
from sklearn import metrics
import numpy as np
from torch.utils.data import Dataset

from torch.utils.data import DataLoader
from tqdm import tqdm 
import gc



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")



@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=True)
    seed: int = field(default=42)

logger = logging.getLogger(__name__)

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
        return dict(input_ids=self.input_ids[i], attention_mask=self.attention_mask[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.Tensor(labels).long()
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )
def clean_gpu():
    torch.cuda.empty_cache()
    gc.collect()

"""
Manually calculate the accuracy, f1, matthews_correlation, precision, recall with sklearn.
"""
def calculate_metric_with_sklearn(logits: np.ndarray, labels: np.ndarray):
    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])
    print(labels.shape)
    probabilities = normalize(logits, axis=1, norm='l1')    
    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    print(valid_mask.shape)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]
    auc = metrics.roc_auc_score(valid_labels, probabilities[valid_mask][:, 1])
    
    return {
        "auc": auc,
        "accuracy": metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
    }


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(logits, labels):
    
    if isinstance(logits, tuple):  # Unpack logits if it's a tuple
        logits = logits[0]
    return calculate_metric_with_sklearn(logits, labels)


def prediction():
    
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    model_class = "BertForSequenceClassification"

    random.seed(training_args.seed)
    np.random.seed(training_args.seed)
    torch.manual_seed(training_args.seed)
    if training_args.n_gpu > 0:
        torch.cuda.manual_seed_all(training_args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # clean_gpu()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True,
    )
    config = BertConfig.from_pretrained(model_args.model_name_or_path,)
  
    dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     kmer=data_args.kmer)
    print("data path: ", data_args.data_path)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    # training_args.output_dir
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        trust_remote_code=True
        
    )
    model.to(device) 
   

    preds = np.zeros([len(dataset),2])

# Create a DataLoader for the training dataset
    
    batch_size = training_args.per_device_eval_batch_size * max(1, training_args.n_gpu)
    train_loader = DataLoader(dataset, batch_size)
    
   
    preds = np.zeros([len(dataset),2])
    out_label_ids = np.zeros([len(dataset)])
   
    softmax = torch.nn.Softmax(dim=1)
# Run the model on the dataset to extract attention scores and predictions
   
    for index, batch in enumerate(tqdm(train_loader, desc="Processing batches")):
        model.eval()
        batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
        with torch.no_grad():
            input_ids = batch['input_ids'].to(training_args.device)
            attention_mask = batch['attention_mask'].to(training_args.device)
            labels = batch['labels'].to(training_args.device)
    
            outputs = model(input_ids,attention_mask=attention_mask, output_attentions=True, return_dict=True)
            
            logits = outputs.logits
            out_label_ids[index*batch_size:index*batch_size+len(batch['input_ids'])] = labels.detach().cpu().numpy()
            preds[index*batch_size:index*batch_size+len(batch['input_ids']),:] = logits.detach().cpu().numpy()
            
    result = {}  
    prob = softmax(torch.tensor(preds, dtype=torch.float32))[:,1].numpy()
    np.save(os.path.join(training_args.output_dir, "pred_results.npy"), prob)
    result = compute_metrics(preds,out_label_ids)
    print(result)   
    print(prob.shape)
    print(prob)

  
    




if __name__ == "__main__":
    prediction()
