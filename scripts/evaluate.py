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
import keras

# Machine Learning
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.callbacks import ModelCheckpoint, EarlyStopping
from utils import generate_data, depth_statistics
from model import build_unet_model
from loss import custom_loss
from keras.models import load_model

# Metadata
BASE_PATH = "./data"
#TRAIN_PATH = os.path.join(BASE_PATH, "nyu2_train")
TEST_PATH = os.path.join(BASE_PATH, "nyu2_test")

#df_metadata_train = pd.read_csv(os.path.join(BASE_PATH, "nyu2_train.csv"), header=None, names=["rgb", "depth"])
#df_metadata_train["scene"] = df_metadata_train["rgb"].apply(lambda x: "_".join(x.split("/")[2].split("_")[:-2]))
df_metadata_test = pd.read_csv(os.path.join(BASE_PATH, "nyu2_test.csv"), header=None, names=["rgb", "depth"])

#df_metadata_train["depth_range"], df_metadata_train["mean"], df_metadata_train["std"] = zip(*df_metadata_train['depth'].progress_apply(depth_statistics))
df_metadata_test["depth_range"], df_metadata_test["mean"], df_metadata_test["std"] = zip(*df_metadata_test['depth'].progress_apply(depth_statistics))

df_metadata_test["rgb"] = "./" + df_metadata_test["rgb"]
df_metadata_test["depth"] = "./" + df_metadata_test["depth"]

test_data = generate_data(df_metadata_test, train=False)

unet_model = build_unet_model()
#unet_model.summary()

# Model Compiling
#unet_model.compile(
#    optimizer=tf.keras.optimizers.Adam(), 
#    loss=custom_loss,
#)

#unet_model = unet_model.load_weights('./model/best_model_bs8_epochs8.h5')

with tf.keras.utils.custom_object_scope({'custom_loss': custom_loss}):
    # Load the model
    unet_model = load_model('./model/best_model_bs8_epochs8.h5')

# Evaluation
unet_scores = unet_model.evaluate(
    test_data, 
    verbose=1
)

print("%s: %.4f" % ("Evaluate Test Custom Loss", unet_scores))