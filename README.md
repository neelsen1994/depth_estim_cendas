# Monocular Depth Estimation with U-Net

This repository contains the code for a monocular depth estimation project, created as part of a recruitment task assigned by Cendas Gmbh. The project utilizes a U-Net architecture for depth estimation.

## Overview

The goal of this project is to implement a monocular depth estimation model using deep learning techniques. First, a simple baseline model is trained with an encoder-decoder architecture. Then, the U-Net architecture is employed for its effectiveness in image-to-image regression tasks. This repository contains scripts for training, testing, and visualizing the output for the monocular depth estimation task using a U-Net architecture, leveraging the NYU Depth V2 dataset.

## Project Structure

- `scripts/`: Directory containing Python scripts.
  - `baseline_model.py`: Python script for creating simple encoder-decoder arcitecture with NYU Depth V2 dataset available in TFDS.   
  - `train.py`: Python script for training the monocular depth estimation model with U-Net and using a subset of the NYU Depth V2 dataset downloaded from [kaggle](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2).
  - `evaluate.py`: Python script for testing the trained model on new data.
  - `utils.py`: Utility functions used in the project.
  - `loss.py`: Implementing custom loss which is a combination of SSIM, L1, and Edge Loss.
  - `model.py`: Implementation of the U-Net model for depth estimation.
  - `visualize.py`: Script for visualizing the results of the depth estimation.
- `Literature/`: Directory containing published papers for bibliographic research work.
- `outputs/`: Directory containing the depth images predicted by the trained models.
- `plots/`: Directory containing the loss curve for training the U-Net model.
- **Assessment Report**: This file contains a brief explanation about each task assigned for the project.

## Usage

### Installation

```bash
git clone https://github.com/neelsen1994/depth_estim_cendas.git
cd depth_estim_cendas
pip install -r requirements.txt
```

### Training

Download the dataset from [here](https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2), create a `data/` directory in the project root directory, and store it there so that it can be easily accessed from the code. The dataset directory structure should be the following:

```
/depth_estim_cendas
  ├── data
  │   ├── nyu2_train
  │   └── nyu2_test
  │   └── nyu2_train.csv
  │   └── nyu2_test.csv
```
To train the model, use the `train.py` script and run the command from the project root directory. You can customize the training parameters by modifying the script. 

```bash
python ./scripts/train.py
```

The trained model is available for download. Create a `model/` directory in the project root directory, and store the downloaded model there so that it can be easily accessed from the evaluation script.

Baseline model: [Download Model](https://drive.google.com/file/d/1Dus3U8t3iR2yiTdQaIcp1j90zG7lolga/view?usp=sharing)

U-Net Model: [Download Model](https://drive.google.com/file/d/1HD05i0DMDgtU0PWUFBssCL867-BWmrR1/view?usp=sharing)

### Testing

For testing the trained model on new data, use the `evaluate.py` script and run the command from the project root directory.

```bash
python ./scripts/evaluate.py
```

### Visualization 

The visualize.py script provides tools for visualizing the results of the depth estimation on sample test images.

```bash
python ./scripts/visualize.py
```

## Acknowledgments

The project was inspired as part of the assessment for the hiring process at Cendas Gmbh. I would like to express my gratitude to Alex Zinelis, Klari and Cendas Gmbh for the opportunity to work on this project.

