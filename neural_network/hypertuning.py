# %%
import numpy as np
from tensorflow import keras
import tensorflow as tf
import keras_tuner as kt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, concatenate, Conv2DTranspose

# %%

# Create dummy input data
bc = np.loadtxt('../simp/results_merge_2/bc.txt')
load = np.loadtxt('../simp/results_merge_2/load.txt')
#vol = np.loadtxt('../simp/results_merge_2/vol.txt')
output = np.loadtxt('../simp/results_merge_2/output.txt')

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))
    #input_data[i, :, :, 2] = load[i].reshape((61,61))

output_train = output.reshape(output.shape[0], 60, 60)

input_test = input_data[-1000:]
output_test = output_train[-1000:]
# %%

def UNN_model(hp):
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
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate), 
                loss='binary_crossentropy',
                metrics=['accuracy'])
    return model


# %%

tuner = kt.Hyperband(UNN_model,
                     objective='val_accuracy',
                     max_epochs=50,
                     factor=3,
                     directory='tunner_unn',
                     project_name='U_NN_tuner')

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner.search(input_test, output_test, epochs=50, validation_split=0.2, callbacks=[stop_early])
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

print(f"""
The hyperparameter search is complete. The optimal number of units in the first densely-connected
layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
is {best_hps.get('learning_rate')}.
""")

