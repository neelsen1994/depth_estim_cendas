import tensorflow as tf
from keras.layers import Input
from keras.models import Model

# Build U-Net Model
def build_unet_model():
    # Inputs
    inputs = Input((480, 640, 3))
    
    # Encoder: Downsample
    conv1, pool1 = downsample_block(inputs, 32)
    conv2, pool2 = downsample_block(pool1, 64)
    conv3, pool3 = downsample_block(pool2, 128)
    conv4, pool4 = downsample_block(pool3, 256)
    
    # Bottleneck
    bottleneck = double_conv_block(pool4, 512)
    
    # Decoder: Upsample
    up6 = upsample_block(bottleneck, conv4, 256)
    up7 = upsample_block(up6, conv3, 128)
    up8 = upsample_block(up7, conv2, 64)
    up9 = upsample_block(up8, conv1, 32)
    
    # Outputs
    outputs = tf.keras.layers.Conv2D(1, (3, 3), padding="same", activation="sigmoid")(up9)
    
    # Model
    unet_model = Model(inputs, outputs, name="U-Net")
    
    return unet_model

# Build Block Model
def double_conv_block(x, n_filters):
    x = tf.keras.layers.Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.Conv2D(n_filters, (3, 3), padding="same", activation="relu")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    return x
    
def downsample_block(x, n_filters):
    conv = double_conv_block(x, n_filters)
    pool = tf.keras.layers.MaxPool2D((2, 2))(conv)
    return conv, pool
    
def upsample_block(x, conv, n_filters):
    x = tf.keras.layers.Conv2DTranspose(n_filters, (3, 3), strides=(2, 2), padding="same")(x)
    x = tf.keras.layers.LeakyReLU(alpha=0.2)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    x = tf.keras.layers.concatenate([x, conv])
    
    x = double_conv_block(x, n_filters)
    return x