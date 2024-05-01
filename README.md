# GRAPE: Graph Neural Network-based RelAtion Prediction for Ecotoxicology

## Overview
This document provides instructions for running the GRAPE project, which involves predictive modelling in ecotoxicology using graph neural networks (GNNs), multi-layer perceptron (MLP), and logistic regression (LR) models.

### Step 0: Pre-processing
- `0_ecotox_preprocess.py`: Initialize file paths and preprocess raw data (envirotox_20230324.csv, downloaded from [EnviroTox Database](https://envirotoxdatabase.org/)).
  - Set the correct file path for 'ecotox_rawdata_file' in `args_parser.py`.
  - Modify filter conditions to customize experiments (e.g., change from 'LC50' to 'EC50').
  - Specify the output path and filename for the pre-processed file 'ecotox_file' in `args_parser.py`.

### Step 1: Data Preparation
- `1_ecotox_data_prep.py`: Prepare data in the desired format using pre-processed files.
  - Set file paths and filenames for saving species (u) features - 'u_filename_raw' and 'u_filename'.
  - Set file paths and filenames for saving chemicals (u) features - 'v_filename_raw' and 'v_filename'.
  - Configure flags (`args.compute_feats`, `args.save_feats`) and threshold values (`args.conc_threshold`).
  - Set the path and filenames for the 'train_file', 'val_file' and 'test_file' in `args_parser.py` to save train, val and test data.

### Step 2: Model Training and Inferencing
- `2_ecotox_GNN_benchmark.py`: Train GNN models or perform inferencing.
  - Set the `name_str` for model naming and other parameters in `args_parser.py`.
  - Specify model parameters, model folder path, and results folder path.
  - Use `train_flag` to train models (set to 0 for inference).
- `2b_ecotox_MLP_benchmark.py` and `2c_ecotox_LR_benchmark.py` for training MLP and LR models.

## Note
Make sure to set file paths, filenames, and parameters correctly before running the scripts.

## Example Usage
```bash
python 0_ecotox_preprocess.py
python 1_ecotox_data_prep.py
python 2_ecotox_GNN_benchmark.py --name_str 19_02_24_ --train_flag 1 
# Or for inference
python 2_ecotox_GNN_benchmark.py --name_str 19_02_24_ --train_flag 0
