# Deep Neural Network for Medium Distribution Diagram Generation

This repository contains a deep neural network designed to generate a medium distribution diagram through a radar B-scan. The model is intended to be deployed on a ground penetration radar (GPR) system that detects defects within concrete walls.

## Overview

The project consists of several key components:

1. **Model Definition**: The neural network model is defined in `train1.py`.
2. **Training and Testing**: The training and testing of the model are handled in `predict1.py`.
3. **Combining Results**: The script `combine.py` is used to combine the generated model outputs with the ground truth data for comparison.

## Loss Functions

Currently, the model uses Mean Squared Error (MSE) and Multi-Scale Structural Similarity Index (MS-SSIM) as loss functions. Given that concrete anomalies occupy a small portion of the scan, we are also experimenting with Focal Loss to improve the detection of these small defects.

## Files Description

- `train1.py`: This file contains the definition of the neural network model (`GPRNet`) and the dataset class (`MyDataset`). It also includes the necessary imports and configurations for training the model.
- `predict1.py`: This file is responsible for training and testing the model. It includes functions for generating pseudo-color images from the model outputs and saving them.
- `combine.py`: This script combines the generated model outputs with the ground truth images for visual comparison.

## Usage

1. **Training the Model**: Run `predict1.py` to train the model using the dataset provided in the `train` directory.
2. **Testing the Model**: The same script, `predict1.py`, can be used to test the model on the test dataset.
3. **Combining Results**: Use `combine.py` to combine the generated model outputs with the ground truth images for comparison.

## Future Work

We are currently testing the effectiveness of Focal Loss in improving the detection of small anomalies within the concrete scans. Further improvements and optimizations will be made based on the results of these experiments.
