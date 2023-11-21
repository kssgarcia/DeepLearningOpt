# %%
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate, Conv2DTranspose

# Create dummy input data
bc = np.loadtxt('results_1f/bc.txt')
load = np.loadtxt('results_1f/load.txt')
output_tensor = np.loadtxt('results_1f/output.txt')

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_tensor = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_tensor[i, :, :, 0] = bc[i].reshape((61,61))
    input_tensor[i, :, :, 1] = load[i].reshape((61,61))

output_tensor = output_tensor.reshape((output_tensor.shape[0],60,60))

input_data = input_tensor[:10]
output_data = output_tensor[:10]

# %%

# Input layer
input_tensor = Input(shape=(61, 61, num_channels))
print("After Initial Convolution:", input_tensor.shape)

# Initial Convolution Layer (No Padding)
initial = Conv2D(32, kernel_size=(2, 2), activation='relu', padding='valid')(input_tensor)
initial = BatchNormalization()(initial)

# Encoding Blocks
def encoding_block(input_layer, filters):
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(input_layer)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    if x.shape[1] == 15:
        encoded = MaxPooling2D((3, 3))(x)
    else:
        encoded = MaxPooling2D((2, 2))(x)
    return encoded

encoded1 = encoding_block(initial, filters=64)
encoded2 = encoding_block(encoded1, filters=128)
encoded3 = encoding_block(encoded2, filters=256)

# Additional Convolution Layers for Feature Maps
x = Conv2D(256, kernel_size=(7, 7), activation='relu', padding='same')(encoded3)
x = BatchNormalization()(x)
x = Conv2D(256, kernel_size=(7, 7), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

# Decoding Blocks
def decoding_block(input_layer, concat_layer, filters):
    x = concatenate([input_layer, concat_layer], axis=-1)
    if x.shape[1] == 5:
        x = Conv2DTranspose(filters, (3, 3), strides=(3,3), activation='relu', padding='same')(x)
    else:
        x = Conv2DTranspose(filters, (2, 2), strides=(2,2), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    return x

decoded1 = decoding_block(x, encoded3, filters=128)
decoded2 = decoding_block(decoded1, encoded2, filters=64)
decoded3 = decoding_block(decoded2, encoded1, filters=32)

# Final Convolution Layers for Element Solution
x = concatenate([decoded3, initial], axis=-1)
x = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(x)
x = BatchNormalization()(x)
x = Conv2D(32, kernel_size=(7, 7), activation='relu', padding='same')(x)
x = BatchNormalization()(x)

# Output layer with Sigmoid activation for binary classification
output_tensor = Conv2D(1, (1, 1), activation='sigmoid')(x)
print("After Final:", output_tensor.shape)

# Create the model
model = Model(inputs=input_tensor, outputs=output_tensor)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(input_data, output_data, epochs=2, batch_size=10)