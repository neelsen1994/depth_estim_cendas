import tensorflow as tf
from keras import layers, models
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the NYU-Depth V2 dataset
dataset_name = 'nyu_depth_v2'
split = 'train'
dataset, info = tfds.load(name=dataset_name, split=split, with_info=True)

print(info)

# Define the model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(480, 640, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(1))  # Output layer for depth prediction

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Prepare the dataset for training
def preprocess(example):
    image = tf.cast(example['image'], tf.float32) / 255.0  # Normalize image
    depth = example['depth'] / 255.0  # Normalize depth
    return image, depth

# Apply preprocessing to the dataset
train_dataset = dataset.map(preprocess).shuffle(1000).batch(32)

# Train the model
model.fit(train_dataset, epochs=5)

# Test the model on a sample image from the dataset
test_example = next(iter(train_dataset.take(1)))
test_image, test_depth = test_example

# Perform depth prediction using the trained model
predicted_depth = model.predict(tf.expand_dims(test_image, axis=0))[0]

# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(test_image.numpy())
plt.title('RGB Image')

plt.subplot(1, 3, 2)
plt.imshow(test_depth.numpy(), cmap='jet')
plt.title('Ground Truth Depth')

plt.subplot(1, 3, 3)
plt.imshow(predicted_depth, cmap='jet')
plt.title('Predicted Depth')

plt.show()
