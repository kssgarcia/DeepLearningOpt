# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors

# Create dummy input data
bc = np.loadtxt('results/bc.txt')
load_x = np.loadtxt('results/load_x.txt')
load_y = np.loadtxt('results/load_y.txt')
output = np.loadtxt('results/output.txt')

# %%
index = 2
plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].matshow(load_x[index].reshape(61, 61), cmap='gray')
ax[1].matshow(-np.flipud(output[index].reshape(60, 60)), cmap='gray')
fig.show()
