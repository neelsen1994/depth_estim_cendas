# Import Libraries

# Warning
import warnings
warnings.filterwarnings("ignore")

# System
import os
import gc
import shutil
import time
import glob

# Main
import random
import numpy as np 
import pandas as pd
import cv2
from tqdm import tqdm
tqdm.pandas()

# Data Visualization
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
#import tensorflow_datasets as tfds
from keras.layers import Input
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from loss import custom_loss
from utils import generate_data, depth_statistics

# Metadata
BASE_PATH = "./data"
TRAIN_PATH = os.path.join(BASE_PATH, "nyu2_train")
TEST_PATH = os.path.join(BASE_PATH, "nyu2_test")

df_metadata_train = pd.read_csv(os.path.join(BASE_PATH, "nyu2_train.csv"), header=None, names=["rgb", "depth"])
df_metadata_train["scene"] = df_metadata_train["rgb"].apply(lambda x: "_".join(x.split("/")[2].split("_")[:-2]))
df_metadata_test = pd.read_csv(os.path.join(BASE_PATH, "nyu2_test.csv"), header=None, names=["rgb", "depth"])

#df_metadata_train["depth_range"], df_metadata_train["mean"], df_metadata_train["std"] = zip(*df_metadata_train['depth'].progress_apply(depth_statistics))
df_metadata_test["depth_range"], df_metadata_test["mean"], df_metadata_test["std"] = zip(*df_metadata_test['depth'].progress_apply(depth_statistics))

df_metadata_test["rgb"] = "./" + df_metadata_test["rgb"]
df_metadata_test["depth"] = "./" + df_metadata_test["depth"]

test_data = generate_data(df_metadata_test, train=False)

# Create Subplots
fig, axs = plt.subplots(3, 2, figsize=(8, 8))

# Data
scenes = list(df_metadata_train["scene"].value_counts().keys()[1:4])
df_metadata_scene = df_metadata_train[df_metadata_train["scene"]==scenes[0]].reset_index(drop=True)
df_metadata_scene = df_metadata_scene.loc[[0, len(df_metadata_scene)-1], :].reset_index(drop=True)
df_metadata_scene

# Plot
for i, scene in enumerate(scenes):
    # Scene
    axs[i, 0].text(0.5, 0.5, scene, ha='center', va='center', fontsize=12)
    axs[i, 0].axis('off')
    
    df_metadata_scene = df_metadata_train[df_metadata_train["scene"]==scene].reset_index(drop=True)
    df_metadata_scene = df_metadata_scene.loc[[0, len(df_metadata_scene)//2, len(df_metadata_scene)-1], :].reset_index(drop=True)
    
    for j in range(3):
        # RGB Image
        rgb_path = "./" + df_metadata_scene.loc[j, "rgb"]
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        axs[i, 0].imshow(rgb_image)
        axs[i, 0].axis('off')
        
        # Depth Image
        depth_path = "./" + df_metadata_scene.loc[j, "depth"]
        depth_image = cv2.imread(depth_path)
        depth_image = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
        depth_image = np.expand_dims(depth_image, -1)
        axs[i, 1].imshow(depth_image, cmap="inferno")
        axs[i, 1].axis('off')

# Title
plt.suptitle("NYU-Depth-V2: RGB Images and Corresponding Depth Maps", x=0.55, y=0.93)

# Show
plt.show()

shuffled_dataset = test_data.shuffle(buffer_size=634)

# Create Subplots
fig, axs = plt.subplots(3, 3, figsize=(8, 8))

# Model
unet_model = tf.keras.models.load_model('./model/best_model_bs8_epochs8.h5', custom_objects={'custom_loss': custom_loss})

# Plot
for i, (rgb, depth) in enumerate(shuffled_dataset.take(3)):
    #j = (i%2)*3
    
    # Predict
    pred = unet_model.predict(rgb, verbose=False)
    
    # RGB Image
    rgb = rgb[0]
    axs[i, 0].imshow(rgb)
    axs[i, 0].axis('off')
    axs[i, 0].set_title("RGB", fontsize=6)
    
    # Ground Truth Image    
    depth = (depth - tf.reduce_min(depth)) / (tf.reduce_max(depth) - tf.reduce_min(depth))    
    depth = depth[0]
    depth = tf.squeeze(depth)
    
    axs[i, 1].imshow(depth, cmap="viridis")
    axs[i, 1].axis('off')
    axs[i, 1].set_title("Ground Truth", fontsize=6)
    
    # Prediction Image
    pred = np.squeeze(pred[0])    
    axs[i, 2].imshow(pred, cmap="viridis")
    axs[i, 2].axis('off')
    axs[i, 2].set_title("Prediction", fontsize=6)

# Title
plt.suptitle("RGB, Ground Truth, and Prediction Images from Test Dataset", x=0.55, y=0.93)

plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# Show
plt.show()