
# Project Overview

We used pre-trained [DNABERT](https://github.com/jerryji1993/DNABERT/tree/master) and [DNABERT-2](https://github.com/MAGICS-LAB/DNABERT_2/tree/main) and fine-tuned models to discriminate replicating origin sequences from non-origin sequences directly from the final embedding of each sequence. Our methodology included extensive data engineering as well as a suite of explainability techniques for model interpretation.

This repository contains the explainability pipelines for each model.
The structure of the repository is organized as following:

## Datasets

This directory contains four engineered datasets derived from OriDB (the S. cerevisiae genome, or budding yeast).
We selected 325 confirmed ORIs, none exceeding 500 bp, as positive instances to match DNABERT’s input size requirement. Shorter ORIs were asymmetrically extended using flanking nucleotides to reach a uniform length of 500 bp.
To explore research questions related to origin base composition, non-origin sequences were subsampled in various ways (as described in the paper), resulting in four datasets.
All datasets share the same positive instances (origin sequences) but differ in their sets of negative instances (non-origin sequences).

To examine whether the proposed DNABERT explainability pipeline generalizes beyond S. cerevisiae, we applied it to a human replication origin dataset. Two datasets designed for this task and are also availabe in dataset directory. They were generated from the dataset originally proposed by iORI-Epi for the origins of DNA replication in the human genome which is publicly available on the corresponding repository https://github.com/linDing-group/iORI-Epi .

In addition, this folder includes all required scripts for constructing the datasets from the original data sources, as well as preprocessing, train/validation/test splitting, and other data preparation steps.

## DNABERT_explainability

Contains the main scripts for attention score visualization and atttention-base fragment selection modules as parts of our proposed motif discovery pipeline to interpret DNABERT predictions.

## DNABERT_modified_finetuning_module
This folder contains a modified version of the original `finetune.py` script from DNABERT. The modifications were introduced to support the requirements of this project and improve usability for large-scale experiments.


## DNABERT-2_explainability

Includes a module to extract attention scores and represent then in 3 mode and corresponding visualization for interpert model performance. Also explainability pipelines for DNABERT-2, including: Perturbation-based explanations and explainaing with Shapley Values. In DNA_TransSHAP pipeline we adopted TranSHAP (https://github.com/enjakokalj/TransSHAP) for DNA sequence classification by DNABERT-2, developing a modified module that enables SHAP-based explanations of DNA sequences tokenized with BPE.


## DNABERT_2_modified_finetuning_module
This folder includes modified versions of the original train.py scripts from DNABERT-2 which you can use for finetuning or training from scratch. The modifications were introduced to support the requirements of this project and to improve reproducibility.