# Import Libraries

# System
import os
import gc
import shutil
import time
import glob

# Warning
import warnings
warnings.filterwarnings("ignore")

# Main
import random
import numpy as np 
import pandas as pd
import cv2
from tqdm import tqdm
tqdm.pandas()

# Data Visualization
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import plotly
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from IPython.display import Image, display, HTML

# Machine Learning
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping

split = 'train[:5%]'

# Load the NYU-Depth V2 dataset
dataset_name = 'nyu_depth_v2'
dataset, info = tfds.load(name=dataset_name, split=split, with_info=True)

# Print dataset information
print(info)