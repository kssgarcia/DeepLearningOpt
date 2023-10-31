# %%
import numpy as np
import random
import matplotlib.pyplot as plt 
from matplotlib import colors
from os import path, makedirs

# Create dummy input data
bc_1f = np.loadtxt('results_2f_3/bc.txt')
load_1f = np.loadtxt('results_2f_3/load.txt')
output_1f = np.loadtxt('results_2f_3/output.txt')

bc_2f = np.loadtxt('results_merge/bc.txt')
load_2f = np.loadtxt('results_merge/load.txt')
output_2f = np.loadtxt('results_merge/output.txt')

'''
bc_2f_2 = np.loadtxt('results_2f_2/bc.txt')
load_2f_2 = np.loadtxt('results_2f_2/load.txt')
output_2f_2 = np.loadtxt('results_2f_2/output.txt')
'''

# Select a random subset of data with a specified number of elements
def select_random_data(bc, load, output, num_elements):
    random_indices = random.sample(range(len(bc)), num_elements)
    random_bc = bc[random_indices]
    random_load = load[random_indices]
    random_output = output[random_indices]
    return random_bc, random_load, random_output

bc_1f, load_1f, output_1f = select_random_data(bc_1f, load_1f, output_1f, int(bc_1f.shape[0]*0.4))
#bc_2f, load_2f, output_2f = select_random_data(bc_2f, load_2f, output_2f, int(bc_2f.shape[0]*0.3))
#bc_2f_2, load_2f_2, output_2f_2 = select_random_data(bc_2f_2, load_2f_2, output_2f_2, int(bc_2f_2.shape[0]*0.3))


bc = np.concatenate((bc_1f, bc_2f), axis=0)
load = np.concatenate((load_1f, load_2f), axis=0)
output = np.concatenate((output_1f, output_2f), axis=0)

# %%

# Save data
dir = './results_merge_2'
if not path.exists(dir): makedirs(dir)
np.savetxt(dir + '/bc.txt', bc, fmt="%.1f")
np.savetxt(dir + '/load.txt', np.array(load), fmt='%s')
np.savetxt(dir + '/output.txt', np.array(output), fmt="%.3f")

# %%
plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].matshow(load[0].reshape(61, 61), cmap='gray')
ax[1].matshow(-np.flipud(output[0].reshape(60, 60)), cmap='gray')
fig.show()