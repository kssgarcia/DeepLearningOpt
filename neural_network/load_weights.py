# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Dense, Flatten
import numpy as np

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = 17690  # Number of samples in each batch

# Create dummy input data
bc = np.loadtxt('results/bc.txt')
load = np.loadtxt('results/load.txt')
output = np.loadtxt('results/output.txt')

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))

output_train = np.zeros((batch_size,3600))
output_train[:] = output

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

# Load the saved weights
model.load_weights('./weights/weights_epoch_07.h5')

# Continue training with the new data
model.fit(input_data, output, epochs=3, batch_size=1)

# Save the model
model.save('models/first')
