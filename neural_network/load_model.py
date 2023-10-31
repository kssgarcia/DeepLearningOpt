# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import tensorflow as tf
from simp_solver.SIMP import *

# Create dummy input data
bc = np.loadtxt('../simp/results_2f_3/bc.txt')
load = np.loadtxt('../simp/results_2f_3/load.txt')
output = np.loadtxt('../simp/results_2f_3/output.txt')

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 2  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load[i].reshape((61,61))

output_train = np.zeros((batch_size,3600))
output_train[:] = output

print("Input shape:", input_data.shape)
print("Output shape:", output_train.shape)

model = tf.keras.models.load_model('../models/third_NN')

y = model.predict(input_data)

# %%
def custom_load(volfrac, r1, c1, r2, c2, l):
    new_input = np.zeros((1,) + input_shape + (num_channels,))
    bc = np.ones((60+1, 60+1)) * volfrac
    bc[:, 0] = 1
    load = np.zeros((60+1, 60+1), dtype=int)
    load[-r1, -c1] = -l
    load[-r2, -c2] = -l

    new_input[0, :, :, 0] = bc
    new_input[0, :, :, 1] = load

    return new_input 

# %%

test_loss, test_accuracy = model.evaluate(input_data, output)
print(test_loss)
# %%

y_custom = model(custom_load(0.6,1,1, 61, 1, 1), False, None)

plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].imshow(np.flipud(np.array(-y[15000]).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
#ax[0].imshow(np.flipud(np.array(y_custom).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[1].matshow(-np.flipud(output[15000].reshape(60, 60)), cmap='gray')
#ax[1].imshow(-optimization(60, 61, 1, 0.6).reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
fig.show()

plt.ion() 
fig,ax = plt.subplots()
ax.matshow(load[15000].reshape(61, 61), cmap='gray')
fig.show()
