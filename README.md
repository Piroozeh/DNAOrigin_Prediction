
**Project Overview

We used pre-trained DNABERT and DNABERT-2 and fine-tuned them to discriminate replicating origin sequences from non-origin sequences directly from the final embedding of each sequence. Our methodology included extensive data engineering as well as a suite of explainability techniques for model interpretation.

This repository contains the explainability pipelines for each model.
The structure of the repository is organized into three main components:

# datasets

This directory contains four engineered datasets derived from OriDB (the Saccharomyces cerevisiae genome, or budding yeast).
We selected 325 confirmed ORIs, none exceeding 500 bp, as positive instances to match DNABERT’s input size requirement.
Shorter ORIs were asymmetrically extended using flanking nucleotides to reach a uniform length of 500 bp.

To explore research questions related to origin base composition, non-origin sequences were subsampled in different ways to construct four datasets.
All datasets share the same positive instances (origin sequences) but differ in their sets of negative instances (non-origin sequences).

# dnabert_explaining

Contains the complete explainability pipeline used to interpret DNABERT predictions through attention-based motif discovery.

# dnabert2_explaining

Includes explainability methods for DNABERT-2, including: Perturbation-based analysis and SHAP (SHapley Additive exPlanations)**