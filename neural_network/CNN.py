# %%
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
import logging
import matplotlib.pyplot as plt 
from matplotlib import colors

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = 17690  # Number of samples in each batch

# Create dummy input data
bc = np.loadtxt('results_1f/bc.txt')
load = np.loadtxt('results_1f/load.txt')
output = np.loadtxt('results_1f/output.txt')

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))

output_train = np.zeros((batch_size,3600))
output_train[:] = output

print("Input shape:", input_data.shape)
print("Output shape:", output_train.shape)

# %%
# Start the timer
start_time = time.perf_counter()

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
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define the checkpoint callback
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath='/weights/weight.h5',
        save_weights_only=True,   # Set to False to save the entire model (including architecture)
        save_freq='epoch'
)

# Train
model.fit(input_data, output, epochs=2, batch_size=10, callbacks=[model_checkpoint_callback])

# Save the model
model.save('../models/time_test')
end_time = time.perf_counter()
trainnig_time = end_time - start_time
print(trainnig_time)

# Save the log
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logcnn2.txt')
logging.info(f"Trainning time: {trainnig_time} seconds.")

# %%
y = model.predict(input_data)
test_loss, test_accuracy = model.evaluate(input_data, output)
print("Output shape:", y.shape)

# %%
'''
plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].imshow(-y[1].reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[1].imshow(-output[1].reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
fig.show()
'''
