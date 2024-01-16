# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import tensorflow as tf
from simp_solver.SIMP import optimization
from models import CNN_model, UNN_model, ViT_model

# Create dummy input data
bc = np.loadtxt('../simp/results_rand_7/bc.txt')
load_x = np.loadtxt('../simp/results_rand_7/load_x.txt')
load_y = np.loadtxt('../simp/results_rand_7/load_y.txt')
UC_x = np.loadtxt('../simp/results_rand_7/UC_x.txt')
UC_y = np.loadtxt('../simp/results_rand_7/UC_y.txt')
output = np.loadtxt('../simp/results_rand_7/output.txt')

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 5  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load_x[i].reshape((61,61))
    input_data[i, :, :, 2] = load_y[i].reshape((61,61))
    input_data[i, :, :, 3] = UC_x[i].reshape((61,61))
    input_data[i, :, :, 4] = UC_y[i].reshape((61,61))

output_train = output.reshape(output.shape[0], 60, 60)
input_train = input_data[:-1000]
output_train = output_train[:-1000]

input_test = input_data[-1000:]
output_test = output_train[-1000:]

# %%
'''
num_classes = 100
input_shape = (61, 61, 2)

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
'''

model = UNN_model((61,61,num_channels))
model.load_weights('../models/best_models/best_unn_rand_8/cp.ckpt')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
test_loss, test_accuracy = model.evaluate(input_test, output_test)

# %%
def custom_load(volfrac, r1, c1, r2, c2, l):
    new_input = np.zeros((1,) + (61,61) + (num_channels,))
    bc = np.ones((60+1, 60+1))
    bc[:, 0] = 1
    load_y = np.zeros((60+1, 60+1), dtype=int)
    load_y[-r1, -c1] = l
    load_y[-r2, -c2] = l
    load_y[-30, -1] = l
    load_x = np.zeros((60+1, 60+1), dtype=int)

    new_input[0, :, :, 0] = bc
    new_input[0, :, :, 1] = load_x
    new_input[0, :, :, 2] = load_y
    
    return new_input 

input_mod = np.concatenate((input_test, custom_load(0.6, 20, 1, 61, 1, 1)), axis=0)
y = model.predict(input_mod)

# %%

y = model.predict(input_train)
index = 5
plt.ion() 
fig,ax = plt.subplots(1,3)
ax[0].imshow(np.flipud(np.array(-y[index]).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
#ax[0].imshow(np.array(y_custom).reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].matshow(-np.flipud(output_test[index].reshape(60, 60)), cmap='gray')
#ax[1].imshow(-np.flipud(optimization(60, 20, 1, 61, 1, 0.6).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].matshow(load_y[index].reshape(61, 61), cmap='gray')
ax[2].set_title('Load point')
ax[2].set_xticks([])
ax[2].set_yticks([])
fig.show()
