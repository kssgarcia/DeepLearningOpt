# %%
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import colors

# Create dummy input data
bc = np.loadtxt('results_dist/bc.txt')
load = np.loadtxt('results_dist/load.txt')
output = np.loadtxt('results_dist/output.txt')

# %%
print(bc.shape)
index = 210
plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].matshow(load[index].reshape(61, 61), cmap='gray')
ax[1].matshow(-np.flipud(output[index].reshape(60, 60)), cmap='gray')
fig.show()
