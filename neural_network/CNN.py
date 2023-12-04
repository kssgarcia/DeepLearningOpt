# %%
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
import logging
import matplotlib.pyplot as plt 
from matplotlib import colors

# Create dummy input data
bc = np.loadtxt('../simp/results_merge_2/bc.txt')
load = np.loadtxt('../simp/results_merge_2/load.txt')
output = np.loadtxt('../simp/results_merge_2/output.txt')

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))
    #input_data[i, :, :, 2] = vol[i].reshape((61,61))

output_data = output.reshape((output.shape[0],60,60))

input_train = input_data[:-1000]
output_train = output_data[:-1000]

input_test = input_data[-1000:]
output_test = output_data[-1000:]

# %%
# Start the timer
start_time = time.perf_counter()

def CNN_model():
    model = Sequential()

    # Layers down-sampling
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(61, 61, num_channels)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D((2, 2)))

    # Upsampling layers
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(UpSampling2D((2, 2)))

    # Flatten layer
    model.add(Flatten())

    # Output layer
    model.add(Dense(3600, activation='sigmoid'))
    return model

model = CNN_model()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    './best/cp.ckpt',
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose= 1,
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(input_train, input_train, epochs=5, batch_size=10,validation_split=0.1, callbacks=[checkpoint_callback])

# Save the model
model.save('../models/model_unet')
