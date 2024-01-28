# %% 
import numpy as np
import matplotlib.pyplot as plt 

x1 = np.loadtxt('../simp/result_matlab/x_dataL.txt')
load_x1 = np.loadtxt('../simp/result_matlab/load_x_dataL.txt')
load_y1 = np.loadtxt('../simp/result_matlab/load_y_dataL.txt')
bc1 = np.loadtxt('../simp/result_matlab/bc_dataL.txt')

x2 = np.loadtxt('../simp/result_matlab/x_dataL2.txt')
load_x2 = np.loadtxt('../simp/result_matlab/load_x_dataL2.txt')
load_y2 = np.loadtxt('../simp/result_matlab/load_y_dataL2.txt')
bc2 = np.loadtxt('../simp/result_matlab/bc_dataL2.txt')

x = np.concatenate((x1, x2), axis=1).T
load_x = np.concatenate((load_x1, load_x2), axis=1).T
load_y = np.concatenate((load_y1, load_y2), axis=1).T
bc = np.concatenate((bc1, bc2), axis=1).T

print(x.shape)
print(load_x.shape)
print(load_y.shape)
print(bc.shape)

# Generate random input data
input_shape = (61, 61)  # Input size of 61x61
num_channels = 3  # Number of channels in each input array
batch_size = bc.shape[0]  # Number of samples in each batch

input_data = np.zeros((batch_size,) + input_shape + (num_channels,))
for i in range(batch_size):
    input_data[i, :, :, 0] = bc[i].reshape((61,61))
    input_data[i, :, :, 1] = load_x[i].reshape((61,61))
    input_data[i, :, :, 1] = load_y[i].reshape((61,61))

output_data = x.reshape((x.shape[0],60,60))

input_train = input_data[:-1000]
output_train = output_data[:-1000]

input_test = input_data[-1000:]
output_test = output_data[-1000:]

batch_size = input_train.shape[0]
print(input_train.shape, input_test.shape)
print(output_train.shape, output_test.shape)

# %%

index = 15000

plt.ion() 
fig,ax = plt.subplots(2,2)
ax[0,0].matshow(-x[index].reshape(60, 60).T, cmap='gray')
ax[0,1].matshow(-load_x[index].reshape(61, 61).T, cmap='gray')
ax[1,0].matshow(-load_y[index].reshape(61, 61).T, cmap='gray')
ax[1,1].matshow(-bc[index].reshape(61, 61).T, cmap='gray')
fig.show()
