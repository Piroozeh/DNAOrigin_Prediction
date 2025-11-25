
# Project Overview

We used pre-trained DNABERT and DNABERT-2 and fine-tuned models to discriminate replicating origin sequences from non-origin sequences directly from the final embedding of each sequence. Our methodology included extensive data engineering as well as a suite of explainability techniques for model interpretation.

This repository contains the explainability pipelines for each model.
The structure of the repository is organized as following:

## Datasets

This directory contains four engineered datasets derived from OriDB (the S. cerevisiae genome, or budding yeast).
We selected 325 confirmed ORIs, none exceeding 500 bp, as positive instances to match DNABERT’s input size requirement. Shorter ORIs were asymmetrically extended using flanking nucleotides to reach a uniform length of 500 bp.

To explore research questions related to origin base composition, non-origin sequences were subsampled in various ways (as described in the paper), resulting in four datasets.
All datasets share the same positive instances (origin sequences) but differ in their sets of negative instances (non-origin sequences).

## DNABERT_explainability

Contains the complete explainability pipeline used to interpret DNABERT predictions through attention-based motif discovery.

## DNABERT-2_explainability

Includes explainability pipelines for DNABERT-2, including: Perturbation-based explanation and Explainaing with Shapley Values. We adopted TranSHAP (https://github.com/enjakokalj/TransSHAP) for DNA sequence classification by
DNABERT-2, developing a modified module that enables SHAP-based explanations of DNA sequences tokenized with BPE.