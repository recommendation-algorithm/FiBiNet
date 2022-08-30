# FiBiNet++:Improving FiBiNet by Effectively Reducing Model Parameters for CTR Prediction

Code of FiBiNet++:Improving FiBiNet by Effectively Reducing Model Parameters for CTR Prediction.

Also support FiBiNet of paper [FiBiNET: Combining Feature Importance and Bilinear feature Interaction for Click-Through Rate Prediction](https://arxiv.org/abs/1905.09433) .

## Prerequisites

- Python == 3.6
- TensorFlow-GPU == 1.14

## Getting Started

### Installation

- Install TensorFlow-GPU 1.14

- Clone this repo

### Dataset

- Links of datasets are:

  - https://www.kaggle.com/c/criteo-display-ad-challenge
  - https://www.kaggle.com/c/avazu-ctr-prediction
  
- You can download the original datasets and preprocess them by yourself. Run `python -u -m fibinet.preprocessing.{dataset_name}.{dataset_name}_process` to preprocess the datasets. `dataset_name` can be `criteo` or `avazu`. 

- This repo also contains a demo dataset of criteo, which contains 100,000 samples and has been preprocessed. It is used to help demonstrate FiBiNet++.

### Training

You can use `python -u -m fibinet.run --version {version} --config {config_path}` to train a specific model on a dataset. 

Some important parameters are list below, and other hyper-parameters can be found in the code. 

  - version: model version, supports `v1`, `++`, and `custom`, default to `++`. For `custom`, you can adjust all parameter values flexibly.
  - config_path: specifies the paths of the input/output files and the fields of the dataset. It is generated during dataset preprocessing. Support values: `./config/criteo/config_dense.json`, `./config/avazu/config_sparse.json`.
  - mode: running mode, supports `train`, `retrain`, `test`. 

## Acknowledgement

Part of the code comes from [DeepCTR](https://github.com/shenweichen/DeepCTR).

