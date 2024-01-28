import tensorflow as tf
from keras import layers, models, callbacks
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

# Load the NYU-Depth V2 dataset
train_dataset_name = 'nyu_depth_v2'
test_dataset_name = 'nyu_depth_v2'
train_split = 'train[:40000]'
val_split = 'train[40000:]'
test_split = 'validation'

train_dataset, train_info = tfds.load(name=train_dataset_name, split=train_split, with_info=True)
val_dataset, val_info = tfds.load(name=train_dataset_name, split=val_split, with_info=True)
test_dataset, test_info = tfds.load(name=test_dataset_name, split=test_split, with_info=True)

# Define the model for depth estimation
def enc_dec_model():
    inputs = tf.keras.Input(shape=(480, 640, 3))
    
    # Encoder
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    # Bottleneck
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(conv3)

    # Decoder
    up4 = layers.UpSampling2D((2, 2))(conv3)
    up4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up4)
    up4 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(up4)

    up5 = layers.UpSampling2D((2, 2))(up4)
    up5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up5)
    up5 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(up5)

    outputs = layers.Conv2D(1, (1, 1), activation='linear')(up5)  # Output layer with linear activation for regression

    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Create the Encoder-Decoder model
model = enc_dec_model()

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Define a ModelCheckpoint callback to save the best model during training
checkpoint_callback = callbacks.ModelCheckpoint(
    filepath='best_model_baseline.h5',
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# Prepare the training dataset
def preprocess_train(example):
    image = tf.cast(example['image'], tf.float32) / 255.0  # Normalize image
    depth = example['depth'] / 10.0  # Normalize depth from meters to a range of 0 to 1
    return image, depth

# Apply preprocessing to the training dataset
train_dataset = train_dataset.map(preprocess_train).shuffle(1000).batch(32)
val_dataset = val_dataset.map(preprocess_train).shuffle(1000).batch(32)

# Train the model
model.fit(train_dataset, epochs=5, validation_data=val_dataset, callbacks=[checkpoint_callback])

# Load the best model from the saved checkpoint
best_model = tf.keras.models.load_model('best_model_baseline.h5')


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

# Perform depth prediction using the trained model
predicted_depth = best_model.predict(tf.expand_dims(test_image, axis=0))
# Display the results
plt.figure(figsize=(12, 6))

plt.subplot(1, 3, 1)
plt.imshow(test_image.numpy())
plt.title('RGB Image')

plt.subplot(1, 3, 2)
plt.imshow(test_depth.numpy(), cmap='viridis')
plt.title('Ground Truth Depth')

plt.subplot(1, 3, 3)
plt.imshow(predicted_depth.squeeze(), cmap='viridis')  # Remove singleton dimension
plt.title('Predicted Depth')

plt.show()
