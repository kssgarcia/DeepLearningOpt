# %%
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models import ViT_model
import matplotlib.pyplot as plt
from os import path, makedirs

x1 = np.loadtxt('../matlab_simp/x_dataL.txt')
load_x1 = np.loadtxt('../matlab_simp/load_x_dataL.txt')
load_y1 = np.loadtxt('../matlab_simp/load_y_dataL.txt')
vol1 = np.loadtxt('../matlab_simp/vol_dataL.txt')
bc1 = np.loadtxt('../matlab_simp/bc_dataL.txt')

x2 = np.loadtxt('../matlab_simp/x_dataL2.txt')
load_x2 = np.loadtxt('../matlab_simp/load_x_dataL2.txt')
load_y2 = np.loadtxt('../matlab_simp/load_y_dataL2.txt')
vol2 = np.loadtxt('../matlab_simp/vol_dataL2.txt')
bc2 = np.loadtxt('../matlab_simp/bc_dataL2.txt')

x3 = np.loadtxt('../matlab_simp/x_dataL3.txt')
load_x3 = np.loadtxt('../matlab_simp/load_x_dataL3.txt')
load_y3 = np.loadtxt('../matlab_simp/load_y_dataL3.txt')
vol3 = np.loadtxt('../matlab_simp/vol_dataL3.txt')
bc3 = np.loadtxt('../matlab_simp/bc_dataL3.txt')

x = np.concatenate((x1, x2, x3), axis=1).T
load_x = np.concatenate((load_x1, load_x2, load_x3), axis=1).T
load_y = np.concatenate((load_y1, load_y2, load_y3), axis=1).T
vol = np.concatenate((vol1, vol2, vol3), axis=1).T
bc = np.concatenate((bc1, bc2, bc3), axis=1).T

input_shape = (61, 61)  # Input size of 61x61
num_channels = 4  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = vol[i].reshape((61,61))
    input_data[i, :, :, 2] = load_x[i].reshape((61,61))
    input_data[i, :, :, 3] = load_y[i].reshape((61,61))
output_data = x.reshape((x.shape[0],60,60))

input_train = input_data[:-1000]
output_train = output_data[:-1000]

input_val = input_data[-1000:]
output_val = output_data[-1000:]

batch_size = input_train.shape[0]

input_shape = (61, 61, num_channels)

learning_rate = 0.001
weight_decay = 0.0001
image_size = 60  # We'll resize input images to this size
patch_size = 10  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 15
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 20

model = ViT_model(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    './best_matlab_vit/cp.ckpt',
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    save_weights_only=True,
    verbose= 1,
)

earlyStopping_callback = keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    mode="max",
    patience=5,
    verbose=1,
)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(input_train, output_train, epochs=50, batch_size=32, validation_data=(input_val, output_val), callbacks=[checkpoint_callback])

# Save the model
model.save('../models/vit_matlab')

dir = './plots'
if not path.exists(dir): makedirs(dir)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_matlab_vit.png')  # Save the plot as an image
plt.show()

# Plotting training and validation accuracy
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/accuracy_matlab_vit.png')  # Save the plot as an image
plt.show()
