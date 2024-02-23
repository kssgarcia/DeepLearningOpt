# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors
import tensorflow as tf
from simp_solver.SIMP import optimization
from models import CNN_model, UNN_model, ViT_model, PVT_model

x1 = np.loadtxt('../simp/results_matlab/x_dataL.txt')
load_x1 = np.loadtxt('../simp/results_matlab/load_x_dataL.txt')
load_y1 = np.loadtxt('../simp/results_matlab/load_y_dataL.txt')
vol1 = np.loadtxt('../simp/results_matlab/vol_dataL.txt')
bc1 = np.loadtxt('../simp/results_matlab/bc_dataL.txt')

x2 = np.loadtxt('../simp/results_matlab/x_dataL2.txt')
load_x2 = np.loadtxt('../simp/results_matlab/load_x_dataL2.txt')
load_y2 = np.loadtxt('../simp/results_matlab/load_y_dataL2.txt')
vol2 = np.loadtxt('../simp/results_matlab/vol_dataL2.txt')
bc2 = np.loadtxt('../simp/results_matlab/bc_dataL2.txt')

x = np.concatenate((x1, x2), axis=1).T
load_x = np.concatenate((load_x1, load_x2), axis=1).T
load_y = np.concatenate((load_y1, load_y2), axis=1).T
vol = np.concatenate((vol1, vol2), axis=1).T
bc = np.concatenate((bc1, bc2), axis=1).T

input_shape = (61, 61)  # Input size of 61x61
num_channels = 4  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = vol[i].reshape((61,61))
    input_data[i, :, :, 2] = load_x[i].reshape((61,61))
    input_data[i, :, :, 3] = load_y[i].reshape((61,61))
output_data = x.reshape((x.shape[0],60,60))

input_train = input_data[:-1000]
output_train = output_data[:-1000]

input_val = input_data[-1000:]
output_val = output_data[-1000:]

batch_size = input_train.shape[0]

# %%

from models import CNN_model, UNN_model, ViT_model, PVT_model
input_shape = (61, 61, num_channels)

learning_rate = 0.001
weight_decay = 0.0001
num_epochs = 100
num_heads = 12

model = PVT_model(input_shape, num_heads)
model.load_weights('../models/best_models/best_matlab_PVT/cp.ckpt')
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# %%
test_loss, test_accuracy = model.evaluate(input_val, output_val)

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

input_mod = np.concatenate((input_train[:1000], custom_load(0.6, 20, 1, 61, 1, 1)), axis=0)
y = model.predict(input_mod)

# %%

y = model.predict(input_val)

# %%
index = 100
plt.ion() 
fig,ax = plt.subplots(1,3)
ax[0].imshow(np.flipud(np.array(-y[index]).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
#ax[0].imshow(np.array(y_custom).reshape(60, 60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[0].set_xticks([])
ax[0].set_yticks([])
ax[1].matshow(-np.flipud(output_val[index].reshape(60, 60)), cmap='gray')
#ax[1].imshow(-np.flipud(optimization(60, 20, 1, 61, 1, 0.6).reshape(60, 60)), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
ax[1].set_xticks([])
ax[1].set_yticks([])
ax[2].matshow(load_y[index].reshape(61, 61), cmap='gray')
ax[2].set_title('Load point')
ax[2].set_xticks([])
ax[2].set_yticks([])
fig.show()
