# %%
import numpy as np
import matplotlib.pyplot as plt 

# Create dummy input data
bc = np.loadtxt('results_rand_test/bc.txt')
load_x = np.loadtxt('results_rand_test/load_x.txt')
load_y = np.loadtxt('results_rand_7/load_y.txt')
output = np.loadtxt('results_rand_7/output.txt')

# %%
index = 20000
plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].matshow(load_y[index].reshape(61, 61), cmap='gray')
ax[1].matshow(-np.flipud(output[index].reshape(60, 60)), cmap='gray')
fig.show()
