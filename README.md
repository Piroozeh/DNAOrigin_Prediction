
# Project Overview

We used pre-trained [DNABERT](https://github.com/jerryji1993/DNABERT/tree/master) and [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2/tree/main) and fine-tuned models to discriminate replicating origin sequences from non-origin sequences directly from the final embedding of each sequence. Our methodology included extensive data engineering as well as a suite of explainability techniques for model interpretation.

This repository contains the explainability pipelines for each model, as well as fine-tuning pipelines, the SHAP analysis scripts, and the data pre-processing workflows.
The structure of the repository is organized as following:

## Datasets

This directory contains four engineered datasets derived from [OriDB](https://cerevisiae.oridb.org/) for the S. cerevisiae genome, or budding yeast.
We selected 325 confirmed ORIs, none exceeding 500 bp, as positive instances to match DNABERT’s input size requirement. Shorter ORIs were asymmetrically extended using flanking nucleotides to reach a uniform length of 500 bp.
To explore research questions related to origin base composition, non-origin sequences were subsampled in various ways (as described in the paper), resulting in four datasets.
All datasets share the same positive instances (origin sequences) but differ in their sets of negative instances (non-origin sequences).

To examine whether the DNABERt can discriminate origin sequences beyond S. cerevisiae and if proposed DNABERT explainability pipeline generalizes, we applied it to a human replication origin dataset. Two datasets designed for this task and are also availabe in dataset directory. They were generated from the dataset originally proposed by iORI-Epi for the origins of DNA replication in the human genome which is publicly available on the corresponding repository https://github.com/linDing-group/iORI-Epi .

In addition, this folder includes all required scripts for constructing the datasets from the original data sources, as well as preprocessing, train/validation/test splitting, and other data preparation steps.

### Statistical comparison for ACS-Neg
The corresponding scripts are located in the folder `datasets/scripts`. The workflow is as follows:
1. `convert_to_fasta.py` merges train.tsv, dev.tsv, test.tsv and writes all_pos.fa / all_neg.fa.
2. Run Homer script on the negative instances against the motif file template, in order to get motif scores `findMotifs.pl all_neg.fa fasta acs_neg_out/ -find acs_motif_matrix.motif > acs_neg.scan.txt`
3. find_best_score.py extracts the best MotifScore per sequence into acs_pos.best.csv / acs_neg.best.csv.
4. distribution_comparison_homer.py after randomly subsampling negatives from acs_neg.best.csv, run it to check whether positive and negative MotifScore distributions are significantly different.

## DNABERT_explainability

Contains the main scripts for attention score visualization and atttention-base fragment selection modules as parts of our proposed motif discovery pipeline to interpret DNABERT predictions.

## DNABERT_modified_finetuning_module
This folder contains a modified version of the original `finetune.py` script from DNABERT. The modifications were introduced to support the requirements of this project and improve usability for large-scale experiments.


## DNABERT-2_explainability

Includes a module to extract attention scores and represent then in 3 mode and corresponding visualization for interpert model performance. Also explainability pipelines for DNABERT-2, including: Perturbation-based explanations and explainaing with Shapley Values. In DNA_TransSHAP pipeline we adopted TranSHAP (https://github.com/enjakokalj/TransSHAP) for DNA sequence classification by DNABERT-2, developing a modified module that enables SHAP-based explanations of DNA sequences tokenized with BPE.

### AT-index

The AT-index workflow quantifies origin-like AT-rich sequence composition independently of model attribution scores. It is implemented in `DNABERT_2_explainability/AT_index` with two main scripts:
1. `check_SHAP_1.py` computes per-sequence AT-richness features and the standardized AT-index.
2. `plot_SHAP.py performs group-wise statistical comparisons using the AT-index values.


## DNABERT_2_modified_finetuning_module
This folder includes modified versions of the original train.py scripts from DNABERT-2 which you can use for finetuning or training from scratch. The modifications were introduced to support the requirements of this project and to improve reproducibility.
