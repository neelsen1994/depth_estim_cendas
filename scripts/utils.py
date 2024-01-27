import numpy as np 
import pandas as pd
import cv2
import tensorflow as tf

def depth_statistics(image_path):
    img = cv2.imread("./" + image_path)
    depth_range = img.max() - img.min()
    mean = np.mean(img)
    std = np.std(img)
    return depth_range, mean, std

def preprocess_data(rgb_path, depth_path, train):    
    # Read Image
    rgb_image = tf.io.read_file(rgb_path)
    rgb_image = tf.image.decode_jpeg(rgb_image, channels=3)
    depth_image = tf.io.read_file(depth_path)
    depth_image = tf.image.decode_jpeg(depth_image, channels=1)
    
    # Resize
    rgb_image = tf.image.resize(rgb_image, (480, 640))
    depth_image = tf.image.resize(depth_image, (480, 640))
    
    # Flipping
    if(train):
        if(tf.random.uniform(()) > 0.5):
            rgb_image = tf.image.flip_left_right(rgb_image)
            depth_image = tf.image.flip_left_right(depth_image)
    
    # Normalize
    rgb_image = tf.cast(rgb_image, tf.float32) / 255.
    depth_image = tf.cast(depth_image, tf.float32) / 255.
    
    # Expand Dimensions
    depth_image = tf.expand_dims(depth_image, axis=-1)
    
    return rgb_image, depth_image

def generate_data(df, train):
    rgb_paths = df["rgb"].values
    depth_paths = df["depth"].values
    
    data = tf.data.Dataset.from_tensor_slices((rgb_paths, depth_paths))
    data = data.map(lambda x, y: preprocess_data(x, y, train))
    
    if(train):
        data = data.batch(8)
    else:
        data = data.batch(1)
    
    return data