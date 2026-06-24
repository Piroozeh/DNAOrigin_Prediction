## 1.  Modifications to DNABERT-2 train.py Scripts

This folder includes modified versions of the original train.py scripts from DNABERT-2 which you can use for finetuning or training from scratch. The modifications were introduced to support the requirements of this project and to improve reproducibility.

### 1. Training from Scratch Support

Additional scripts and functionality were added to enable training DNABERT-2 models from scratch, in addition to the standard fine-tuning workflow provided in the original implementation.

### 2. Optional Frozen Backbone Training

The training pipeline was modified to provide an option to freeze all DNABERT-2 model parameters except the final classification layer. 
### Code Annotations

All modifications introduced for this project are explicitly marked in the source code using the following comment:

```python
# added by Piroozeh
```

These annotations are intended to make it straightforward to identify project-specific changes relative to the original implementation.

### Original DNABERT-2 Implementation

The original train.py script can be found in the official DNABERT-2 repository:
The original implementation can be found in the [DNABERT-2 repository](https://github.com/MAGICS-LAB/DNABERT_2/tree/main/finetune)


Please refer to the original repository for the unmodified implementation, installation instructions, and additional documentation.



## 2. Prediction Script

A standalone `prediction.py` script is provided for loading a trained model and performing inference on new datasets.

Example usage:

```bash

model_ft=/path/to/model

export DATA_PATH=/path/to/data
export MAX_LENGTH=125

python prediction.py \
    --model_name_or_path ${model_ft} \
    --data_path ${DATA_PATH} \
    --kmer -1 \
    --run_name DNABERT2_${DATA_PATH} \
    --model_max_length ${MAX_LENGTH} \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 16 \
    --fp16 \
    --output_dir ${DATA_PATH} \
    --overwrite_output_dir True \
    --evaluation_strategy steps
```

The script loads a fine-tuned DNABERT-2 model and generates predictions for the provided dataset using the specified configuration.

