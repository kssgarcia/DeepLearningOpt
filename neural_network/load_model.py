# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import tensorflow as tf

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = 3660  # Number of samples in each batch

# Create dummy input data
bc = np.loadtxt('results_multi/bc.txt')
load = np.loadtxt('results_multi/load.txt')
output = np.loadtxt('results_multi/output.txt')

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))

output_train = np.zeros((batch_size,3600))
output_train[:] = output

print("Input shape:", input_data.shape)
print("Output shape:", output_train.shape)

model = tf.keras.models.load_model('../../models/model_CNN2')

# %%
def custom_load(volfrac, r, c, l):
    new_input = np.zeros((1,) + input_shape + (num_channels,))
    bc = np.ones((60+1, 60+1)) * volfrac
    bc[:, 0] = 1
    load = np.zeros((60+1, 60+1), dtype=int)
    load[-r, -c] = l
    #load[-61, -c] = -l

    new_input[0, :, :, 0] = bc
    new_input[0, :, :, 1] = load

    return new_input 

# %%

y = model(custom_load(0.6,61,1, -1), False, None)

plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].imshow(np.array(-y[0]).reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[1].imshow(-output[313].reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
fig.show()
