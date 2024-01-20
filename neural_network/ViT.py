# %%
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from models import ViT_model
import matplotlib.pyplot as plt
from os import path, makedirs

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

output_data = output.reshape((output.shape[0],60,60))

input_train = input_data[:-1000]
output_train = output_data[:-1000]

input_test = input_data[-1000:]
output_test = output_data[-1000:]

batch_size = input_train.shape[0]

# %%

input_shape = (61, 61, num_channels)

learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 100
image_size = 60  # We'll resize input images to this size
patch_size = 10  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 12
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 15

model = ViT_model(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_units, transformer_layers)

checkpoint_callback = keras.callbacks.ModelCheckpoint(
    './best/cp.ckpt',
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

optimizer = keras.optimizers.AdamW(
    learning_rate=learning_rate, weight_decay=weight_decay
)

def pixel_accuracy(y_true, y_pred):
    y_true_binary = tf.round(y_true)
    y_pred_binary = tf.round(y_pred)
    pixel_accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_binary, y_pred_binary), dtype=tf.float32))
    return pixel_accuracy

model.compile(optimizer=optimizer, loss=keras.losses.BinaryCrossentropy(), metrics=['accuracy', pixel_accuracy])
history = model.fit(input_test, output_test, epochs=10, batch_size=32, validation_split=0.1, callbacks=[checkpoint_callback])

# %%
dir = './plots'
if not path.exists(dir): makedirs(dir)
plt.figure(figsize=(10, 6))
plt.semilogy(history.history['loss'][1:], label='Training Loss')
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('ViT Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_plot_sparse.png')  # Save the plot as an image
plt.show()


# Plotting training and validation accuracy
plt.figure(figsize=(10, 6))
plt.semilogy(1-np.array(history.history['accuracy'])[1:], label='Training Accuracy')
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('ViT Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/accuracy_plot_sparse.png')  # Save the plot as an image
plt.show()

