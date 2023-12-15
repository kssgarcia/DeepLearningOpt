# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import tensorflow as tf
from simp_solver.SIMP import optimization

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
model = tf.keras.models.load_model('../models/U_NN2')
#model = tf.keras.models.load_model('../models/vit_last_100')
model.summary()

# %%
test_loss, test_accuracy = model.evaluate(input_test, output_test)
print(test_loss)

# %%
y = model.predict(input_test)

# %%
def custom_load(volfrac, r1, c1, r2, c2, l):
    new_input = np.zeros((1,) + input_shape + (num_channels,))
    bc = np.ones((60+1, 60+1)) * volfrac
    bc[:, 0] = 1
    load = np.zeros((60+1, 60+1), dtype=int)
    load[-r1, -c1] = l
    load[-r2, -c2] = -l
    print(load)

    new_input[0, :, :, 0] = bc
    new_input[0, :, :, 1] = load

    return new_input 

y_custom = model(custom_load(0.6, 1, 60, 61, 60, 1))

index = 300
plt.ion() 
fig,ax = plt.subplots(1,3)
#ax[0].imshow(np.flipud(np.array(-y[index]).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].imshow(np.flipud(np.array(y_custom).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[0].set_xticks([])
ax[0].set_yticks([])
#ax[1].matshow(-np.flipud(output_test[index].reshape(60, 60)), cmap='gray')
ax[1].imshow(-optimization(60, 61, 1, 0.6).reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].matshow(load[index].reshape(61, 61), cmap='gray')
ax[2].set_title('Load point')
ax[2].set_xticks([])
ax[2].set_yticks([])
fig.show()
