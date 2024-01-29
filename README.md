# Monocular Depth Estimation with U-Net

This repository contains the code for a monocular depth estimation project, created as part of a recruitment task assigned by Cendas Gmbh. The project utilizes a U-Net architecture for depth estimation.

## Overview

The goal of this project is to implement a monocular depth estimation model using deep learning techniques. First, a simple baseline model is trained with an encoder-decoder architecture. Then, the U-Net architecture is employed for its effectiveness in image-to-image regression tasks. This repository contains scripts for training, testing, and visualizing the output for the monocular depth estimation task using a U-Net architecture.

## Project Structure

- [`train.py`](scripts/train.py): Python script for training the depth estimation model.
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
```

The trained model is available for download.

Baseline model: [Download Model](https://drive.google.com/file/d/1Dus3U8t3iR2yiTdQaIcp1j90zG7lolga/view?usp=sharing)

U-Net Model: [Download Model](https://drive.google.com/file/d/1HD05i0DMDgtU0PWUFBssCL867-BWmrR1/view?usp=sharing)

## Acknowledgments

The project was inspired by 

