# 4248-project

## Overview 

This project was done as part of CS4248: Natural Language Processing, a course in National University of Singapore (NUS).

Our project aims to analyze the shortcomings and performance of baseline models for the extended Stanford Natural Language Inference dataset [e-SNLI](https://github.com/OanaMariaCamburu/e-SNLI), as well as Facebook's [InferSent](https://github.com/facebookresearch/InferSent), and subsequently propose and implement some areas of improvements based on the findings. The project relates to textual entailment and focuses on two main tasks: (1) generation of explanations from given pairs of premise and hypothesis, and (2) prediction of a label based on the premise, hypothesis, and the generated explanation.

### Abstract

[Insert Project Abstract here]

## Navigating this Repository

The main scripts are located in the `e-SNLI` and `InferSent` directories, each containing Python scripts and notebooks on analysis for e-SNLI and InferSent respectively.

The `plots` directory contains various diagrams created from our analysis, and the `pred_outputs` directory contains the predicted outputs generated by the baseline models when run on the test (and sometimes validation) dataset.

## Setup

Make sure that you have Python version 3 or above before running the scripts, as well as the necessary dependencies.

You may refer to, or even run, `sh bin/init.sh` to download necessary models and data.

## Exploratory data analysis (EDA)

All scripts and data related to EDA can be found in `e-SNLI/EDA`.

There are 2 directories in `e-SNLI/EDA`:
1. `pre-run`
- Content in this directory refers to any analysis done before running of pre-trained models. They involve the combined analysis of train, dev and test data, which amounts to a total of 500,000+ rows.

2. `post-run`
- Content in this directory refers to any analysis done after running of pre-trained models on the test dataset, where predictions have been generated.


