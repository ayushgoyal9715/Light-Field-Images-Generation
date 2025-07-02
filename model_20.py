
import tensorflow as tf
from tensorflow.keras import layers, Model


def build_3d_unet(input_shape=(256, 256, 4, 3)):
    inputs = layers.Input(input_shape)
    x = layers.Conv3D(16, (3, 3, 3), padding="same", activation="relu")(inputs)
    x1 = layers.Reshape((256, 256, 4 * 16))(x)  # Reshape to 2D

    skip1 = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x1)  # Save this for later
    x = layers.MaxPool2D((2,2), padding="same")(skip1)

    skip2 = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)  # Save this for later
    x = layers.MaxPool2D((2,2), padding="same")(skip2)

    skip3 = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)  # Save this for later
    x = layers.MaxPool2D((2,2), padding="same")(skip3)

    skip4 = layers.Conv2D(1024, (3,3), padding="same", activation="relu")(x)  # Save this for later
    x = layers.MaxPool2D((2,2), padding="same")(skip4)

    x = layers.Conv2D(2048, (3,3), padding="same", activation="relu")(x)  # Bottleneck

 



    x = layers.Conv2DTranspose(1024, (3,3), strides=(2,2), padding="same", activation="relu")(x)
    x = layers.Concatenate()([x, skip4])  # Skip connection
    x = layers.Conv2D(1024, (3,3), padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(512, (3,3), strides=(2,2), padding="same", activation="relu")(x)
    x = layers.Concatenate()([x, skip3])  # Skip connection
    x = layers.Conv2D(512, (3,3), padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(256, (3,3), strides=(2,2), padding="same", activation="relu")(x)
    x = layers.Concatenate()([x, skip2])  # Skip connection
    x = layers.Conv2D(256, (3,3), padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(128, (3,3), strides=(2,2), padding="same", activation="relu")(x)
    x = layers.Concatenate()([x, skip1])  # Skip connection
    x = layers.Conv2D(128, (3,3), padding="same", activation="relu")(x)

    x = layers.Conv2DTranspose(64, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, (3,3), padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(15, (3,3), padding="same", activation="relu")(x)

    # Reshaping back to 3D
    # x = layers.Reshape((256, 256, 4, 16))(x)
    # x = layers.Conv2D(filters=15, kernel_size=(3, 3), padding="same", activation="relu")(x)
    x = layers.Reshape((256, 256, 5, 3))(x)
    



    model = Model(inputs, x)
    return model


