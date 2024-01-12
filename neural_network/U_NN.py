# %%
import numpy as np
from tensorflow import keras
from models import UNN_model
import matplotlib.pyplot as plt
from os import path, makedirs

# Create dummy input data
bc = np.loadtxt('../simp/results_merge_3/bc.txt')
load = np.loadtxt('../simp/results_merge_3/load.txt')
output = np.loadtxt('../simp/results_merge_3/output.txt')

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

model = UNN_model((61,61,num_channels))

# %%

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

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(input_test, output_test, epochs=10, batch_size=32, validation_split=0.1, callbacks=[checkpoint_callback, earlyStopping_callback])

# Save the model
model.save('../models/model_unet')

# %%
dir = './plots'
if not path.exists(dir): makedirs(dir)
plt.figure(figsize=(10, 6))
plt.semilogy(history.history['loss'][1:], label='Training Loss')
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('plots/loss_plot.png')  # Save the plot as an image
plt.show()


# Plotting training and validation accuracy
plt.figure(figsize=(10, 6))
plt.semilogy(1-np.array(history.history['accuracy'])[1:], label='Training Accuracy')
#plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('plots/accuracy_plot.png')  # Save the plot as an image
plt.show()
