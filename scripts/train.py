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
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import cv2
from tqdm import tqdm
tqdm.pandas()

# Machine Learning
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import generate_data, depth_statistics
from model import build_unet_model
from loss import custom_loss

# Metadata
BASE_PATH = "./data"
TRAIN_PATH = os.path.join(BASE_PATH, "nyu2_train")

df_metadata_train = pd.read_csv(os.path.join(BASE_PATH, "nyu2_train.csv"), header=None, names=["rgb", "depth"])
df_metadata_train["scene"] = df_metadata_train["rgb"].apply(lambda x: "_".join(x.split("/")[2].split("_")[:-2]))

df_metadata_train["depth_range"], df_metadata_train["mean"], df_metadata_train["std"] = zip(*df_metadata_train['depth'].progress_apply(depth_statistics))

# Paths
df_metadata_train["rgb"] = "./" + df_metadata_train["rgb"]
df_metadata_train["depth"] = "./" + df_metadata_train["depth"]

# Split Train Data with SkLearn Library
df_metadata_train, df_metadata_val = train_test_split(
    df_metadata_train, 
    test_size=0.1, 
    stratify=df_metadata_train["scene"], 
    shuffle=True,
    random_state=2023
)

df_metadata_train = df_metadata_train.reset_index(drop=True)
df_metadata_val = df_metadata_val.reset_index(drop=True)

train_data = generate_data(df_metadata_train, train=True)
val_data = generate_data(df_metadata_val, train=True)

for i, (rgb, depth) in enumerate(train_data.take(5)):  # only take the first 5 examples
    print(f"Example {i+1}:")
    print(f"RGB image shape: {rgb.shape}")
    print(f"RGB image pixel range: min={tf.reduce_min(rgb)}, max={tf.reduce_max(rgb)}")
    print(f"Depth image shape: {depth.shape}")
    print(f"Depth image pixel range: min={tf.reduce_min(depth)}, max={tf.reduce_max(depth)}")
    print("---")

unet_model = build_unet_model()
unet_model.summary()

# Model Compiling
unet_model.compile(
    optimizer=tf.keras.optimizers.Adam(), 
    loss=custom_loss,
)

# Callback setup
early_stopping = EarlyStopping(
    monitor="val_loss",
    mode="min",
    verbose=1,
    patience=2
)
model_checkpoint = ModelCheckpoint(
    "best_unet_model.h5", 
    monitor="val_loss", 
    mode="min", 
    verbose=1,
    save_best_only=True
)
callbacks = [early_stopping, model_checkpoint]

# Training the model
start_time = time.time()
unet_history = unet_model.fit(
    train_data,  
    validation_data=val_data,
    epochs=8,
    callbacks=callbacks
)
end_time = time.time()
unet_time = end_time-start_time

loss = unet_history.history['loss']
val_loss = unet_history.history['val_loss']

# Create a DataFrame for easy plotting
df = pd.DataFrame({
    'Epochs': range(1, len(loss) + 1),
    'Training Loss': loss,
    'Validation Loss': val_loss
})

# Set seaborn style
sns.set(style="whitegrid")

# Create line plot
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='Epochs', y='Training Loss', marker='o', label='Training Loss')
sns.lineplot(data=df, x='Epochs', y='Validation Loss', marker='o', label='Validation Loss')

# Set labels and title
plt.xlabel('Epochs')
plt.ylabel('Custom Loss')
plt.title('Training and Validation Loss')

# Show the plot
plt.show()
