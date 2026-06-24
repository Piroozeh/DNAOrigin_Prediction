# Modifications to DNABERT `run_finetune.py`

This folder includes a modified version of the original `finetune.py` script from DNABERT. The modifications were introduced to support the requirements of this project and improve usability for large-scale experiments.

## Summary of Modifications

### 1. Training from Scratch Support

Additional scripts and functionality were added to enable training DNABERT models from scratch, in addition to the standard fine-tuning workflow provided in the original implementation.

### 2. Optional Frozen Backbone Training

The training pipeline was modified to provide an option to freeze all DNABERT model parameters except the final classification layer. This option added in order to investigate the validation loss behaviour for budding yeast datasets which have relatively small size of samples, we trained additional models in which the encoder was frozen and only the classification layer was updated. Results for our downstream task indicate that training only the classification layer (1538 trainable parameters) was insufficient for effective adaptation to the replication-origin prediction task. 

### 3. Training Metrics Logging

The training module was modified to output and record training metrics during model optimization. These changes facilitate monitoring of training progress and simplify downstream analysis of model performance.

### 4. Memory-Optimized Visualization Module

The visualization module was refactored to support attention-score extraction on large datasets with significantly reduced memory requirements.

Key changes include:

* Avoiding storage of full attention tensors for all samples.
* Computing visualization scores directly during batch processing.
* Vectorizing k-mer score aggregation operations.
* Storing only the final per-position scores for [CLS] instead of extracting full attention maps.

These modifications substantially reduce memory consumption and improve processing speed when extracting attention-based visualizations from large datasets, such as k562_Len512 and k562_LenMatch.

## Code Annotations

All modifications introduced for this project are explicitly marked in the source code using one of the following comments:

```python
# added by Piroozeh
```

or

```python
# modified by Piroozeh
```

These annotations are intended to make it easy to identify and review all project-specific changes.

## Original DNABERT Implementation

The original `finetune.py` script can be found in the official DNABERT repository:
[DNABERT repository](https://github.com/jerryji1993/DNABERT/blob/master/examples/run_finetune.py)

Please refer to the original repository for the unmodified implementation and additional documentation.
