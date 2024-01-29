# Monocular Depth Estimation with U-Net

This repository contains scripts for training and testing a monocular depth estimation model using a U-Net architecture.

## Project Structure

- `train.py`: Python script for training the depth estimation model.
- `test.py`: Python script for testing the trained model on new data.
- `utils.py`: Utility functions used in the project.
- `loss.py`: Implementing custom loss which is a combination of SSIM, L1, and Edge Loss.
- `model.py`: Implementation of the U-Net model for depth estimation.
- `visualize.py`: Script for visualizing the results of the depth estimation.

## Usage

### Training

To train the model, use the `train.py` script. You can customize the training parameters by modifying the script.

```bash
python train.py
