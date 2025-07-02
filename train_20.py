import random
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, Model
from PIL import Image
import os
import numpy as np
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.utils import Sequence
from PIL import Image
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import LearningRateScheduler
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.image import ssim
from data_generation_20 import LightFieldDataGenerator
from model_20 import build_3d_unet


import tensorflow as tf
from tensorflow.image import ssim
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from sklearn.model_selection import train_test_split

import tensorflow as tf

def combined_loss(y_true, y_pred, mse_weight=1.0, mae_weight=1.0, ssim_weight=1.0):
    # Initialize MSE and MAE loss functions
    mse = tf.keras.losses.MeanSquaredError()
    mae = tf.keras.losses.MeanAbsoluteError()

    # Compute MSE and MAE over the entire tensor
    mse_loss = mse(y_true, y_pred)
    mae_loss = mae(y_true, y_pred)

    # Compute SSIM for each frame (dimension 3, size 5) and average
    ssim_values = []
    for i in range(5):  # Iterate over the 5 frames
        y_true_frame = y_true[:, :, :, i, :]  # Shape: [batch_size, 256, 256, 3]
        y_pred_frame = y_pred[:, :, :, i, :]  # Shape: [batch_size, 256, 256, 3]
        ssim_frame = tf.image.ssim(y_true_frame, y_pred_frame, max_val=1.0)
        ssim_values.append(ssim_frame)
    
    # Average SSIM across frames
    ssim_loss = 1.0 - tf.reduce_mean(ssim_values)

    # Combine losses with weights
    total_loss = mse_weight * mse_loss + mae_weight * mae_loss + ssim_weight * ssim_loss
    return total_loss


# Define SSIM metric (unchanged from your code)
def ssim_metric(y_true, y_pred):
    # Ensure both tensors are float32
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    
    # Assuming y_true and y_pred are batched image tensors
    ssim_sum = 0.0
    for i in range(batch_size):  # Replace with your actual batch processing logic
        y_true_frame = y_true[i]
        y_pred_frame = y_pred[i]
        ssim_frame = tf.image.ssim(y_true_frame, y_pred_frame, max_val=1.0)
        ssim_sum += ssim_frame
    return ssim_sum / batch_size

# Build model
model = build_3d_unet()
model.summary()

# Define batch size
batch_size = 2
root_path = "/home/ag/lfi/dataset/lofimages"

data_folders = sorted([os.path.join(root_path, d) for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))])
# Create generators for training and validation
train_folders, val_folders = train_test_split(data_folders, test_size=0.2, random_state=42)
train_generator = LightFieldDataGenerator(train_folders, batch_size=batch_size, shuffle=True)
val_generator = LightFieldDataGenerator(val_folders, batch_size=batch_size, shuffle=False)

# Define callbacks
def scheduler(epoch, lr):
    if epoch < 10:
        return lr
    else:
        return lr * 0.95 

checkpoint_callback = ModelCheckpoint(
    "model_weights_epoch_{epoch:02d}.weights.h5",
    save_weights_only=True,
    save_best_only=False,
    verbose=1
)
callback = LearningRateScheduler(scheduler)
def ssim_metric(y_true, y_pred):
    ssim_values = []
    for i in range(5):  # Iterate over the 5 frames
        y_true_frame = y_true[:, :, :, i, :]  # Shape: [batch_size, 256, 256, 3]
        y_pred_frame = y_pred[:, :, :, i, :]  # Shape: [batch_size, 256, 256, 3]
        ssim_frame = tf.image.ssim(y_true_frame, y_pred_frame, max_val=1.0)
        ssim_values.append(ssim_frame)
    return tf.reduce_mean(ssim_values)
# Compile the model with combined loss
model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: combined_loss(y_true, y_pred, mse_weight=1.0, mae_weight=1.0, ssim_weight=1.0),
    metrics=[tf.keras.metrics.MeanSquaredError()]
)

# Train the model
epochs = 300
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=epochs,
    callbacks=[checkpoint_callback, callback]
)

# Save final model weights
model.save_weights("model_weights.h5")
