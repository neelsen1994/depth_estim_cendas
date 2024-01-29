import tensorflow as tf
from keras import layers, models, callbacks
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import os

# Load the NYU-Depth V2 dataset
test_dataset_name = 'nyu_depth_v2'
test_split = 'validation'

test_dataset, test_info = tfds.load(name=test_dataset_name, split=test_split, with_info=True)

# Load the best model from the saved checkpoint
best_model = tf.keras.models.load_model('./model/best_model_baseline.h5')


# Prepare the testing dataset
def preprocess_test(example):
    image = tf.cast(example['image'], tf.float32) / 255.0  # Normalize image
    depth = example['depth'] / 10.0  # Normalize depth from meters to a range of 0 to 1
    return image, depth

# Apply preprocessing to the testing dataset
test_dataset_eval = test_dataset.map(preprocess_test).batch(32)

# Evaluate the best model on the test dataset
test_loss = best_model.evaluate(test_dataset_eval)
print(f'Test Loss of the Best Model: {test_loss}')

test_dataset = test_dataset.map(preprocess_test)

# Test the model on a sample image from the test dataset
test_example = next(iter(test_dataset.take(1)))
test_image, test_depth = test_example

# Prepare the custom testing dataset from the kaggle repository
def preprocess_custom_test(image_path):
    image = Image.open(image_path)
    image = np.array(image) / 255.0  # Normalize image
    image = tf.image.resize(image, (480, 640))  # Resize image to match model input shape
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Path to the custom test images directory
custom_test_directory = './data/nyu2_test/'

# Get a list of image paths in the custom test directory
custom_test_image_paths = [custom_test_directory + file for file in os.listdir(custom_test_directory) if file.endswith('_colors.png')]

# Perform depth prediction using the trained model for each custom test image
for image_path in custom_test_image_paths:
    custom_test_image = preprocess_custom_test(image_path)
    predicted_depth = best_model.predict(custom_test_image)

    # Display the results for each custom test image
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.imshow(custom_test_image.numpy().squeeze())
    plt.title('Custom Test Image')

    plt.subplot(1, 2, 2)
    plt.imshow(predicted_depth.squeeze(), cmap='viridis')  # Remove singleton dimension
    plt.title('Predicted Depth')

    plt.show()
