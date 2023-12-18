# %%
import numpy as np
from tensorflow import keras
from models import UNN_model

# Create dummy input data
bc = np.loadtxt('../simp/results_merge_2/bc.txt')
load = np.loadtxt('../simp/results_merge_2/load.txt')
#stress = np.loadtxt('../simp/results_FEM/stress.txt')
#uc = np.loadtxt('../simp/results_FEM/uc.txt')
#vol = np.loadtxt('../simp/results_FEM/vol.txt')
output = np.loadtxt('../simp/results_merge_2/output.txt')

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))
    #input_data[i, :, :, 2] = stress[i].reshape((61,61))
    #input_data[i, :, :, 3] = uc[i].reshape((61,61))
    #input_data[i, :, :, 3] = vol[i].reshape((61,61))

output_data = output.reshape((output.shape[0],60,60))

input_train = input_data[:-1000]
output_train = output_data[:-1000]

input_test = input_data[-1000:]
output_test = output_data[-1000:]


model = UNN_model(num_channels)

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
model.fit(input_test, output_test, epochs=15, batch_size=32, validation_split=0.1, callbacks=[checkpoint_callback, earlyStopping_callback])

# Save the model
model.save('../models/model_unet')
