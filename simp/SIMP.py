# %%
import time
import logging
import numpy as np
from scipy.sparse.linalg import spsolve
import solidspy.assemutil as ass # Solidspy 1.1.0

import matplotlib.pyplot as plt 
from matplotlib import colors

from beams import *
from SIMP_utils import *

# Start the timer
start_time = time.time()

np.seterr(divide='ignore', invalid='ignore')

# Initialize variables
length = 10
height = 10
nx = 60
ny= 60
niter = 60
penal = 3
Emin=1e-9
Emax=1.0

# Initialize storage variables
input_load = []
input_bc = []
ouput_rho = []

a = True
iter = 0
r = 30
c = 10
volfrac = 0.6
load = np.zeros((nx+1, ny+1), dtype=int)
load[-r, -c] = 1
iter += 1
node_index = nx*r+(r-c)
nodes, mats, els, loads = beam(L=length, H=height, nx=nx, ny=ny, n=node_index)
print(loads)

# Initialize the design variables
change = 10
g = 0
rho = volfrac * np.ones(ny*nx, dtype=float)
sensi_rho = np.ones(ny*nx)
rho_old = rho.copy()
d_c = np.ones(ny*nx)

# Create and initialize channels
bc = np.ones((nx + 1, ny + 1)) * volfrac
bc[:, 0] = 1

if a:
    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4
    centers = center_els(nodes, els)
    E = mats[0,0]
    nu = mats[0,1]
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8])
    kloc = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]);
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8)

a = False

for _ in range(niter):

    # Check convergence
    if change < 0.01:
        print('Convergence reached')
        break

    # Change density 
    mats[:,2] = Emin+rho**penal*(Emax-Emin)

    # System assembly
    stiff_mat = sparse_assem(els, mats, nodes[:, :3], neq, assem_op, kloc)
    rhs_vec = ass.loadasem(loads, bc_array, neq)

    # System solution
    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp)

    # Sensitivity analysis
    sensi_rho[:] = (np.dot(UC[els[:,-4:]].reshape(nx*ny,8),kloc) * UC[els[:,-4:]].reshape(nx*ny,8) ).sum(1)
    d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
    d_c[:] = density_filter(centers, r_min, rho, d_c)

    # Optimality criteria
    rho_old[:] = rho
    rho[:], g = optimality_criteria(nx, ny, rho, d_c, g)

    # Compute the change
    change = np.linalg.norm(rho.reshape(nx*ny,1)-rho_old.reshape(nx*ny,1),np.inf)


plt.ion() 
fig,ax = plt.subplots(1,2)
ax[0].imshow(-rho.reshape(60,60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[0].set_title('Predicted')
ax[1].imshow(-rho.reshape(60,60), cmap='gray', interpolation='none',norm=colors.Normalize(vmin=-1,vmax=0))
ax[1].set_title('Expected')
fig.show()
