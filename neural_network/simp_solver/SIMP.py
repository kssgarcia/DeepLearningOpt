# %%
import time
import numpy as np
import random
from scipy.sparse.linalg import spsolve
import solidspy.assemutil as ass # Solidspy 1.1.0

import matplotlib.pyplot as plt 
from matplotlib import colors

from .beams import *
from .SIMP_utils import *

# Start the timer
start_time = time.time()

np.seterr(divide='ignore', invalid='ignore')

def optimization(n_elem, r1, c1, r2, c2, volfrac):
    # Initialize variables
    length = 60
    height = 60
    nx = n_elem
    ny= n_elem
    niter = 60
    penal = 3 # Penalization factor
    Emin=1e-9 # Minimum young modulus of the material
    Emax=1.0 # Maximum young modulus of the material

    node_index1 = nx*r1+(r1-c1) # Change the linear 
    node_index2 = nx*r2+(r2-c2) # Change the linear 
    node_index3 = nx*30+(30-1) # Change the linear 
    nodes, mats, els, loads = beam(L=length, H=height, nx=nx, ny=ny, n1=node_index1, n2=node_index2, n3=node_index3)
    print(loads)

    '''
    directions = [[0,1], [1,0], [0,-1], [-1,0]]
    num_forces = random.randint(1,5)
    dirs = np.array([random.choice(directions) for _ in range(num_forces)])
    positions = np.array([[random.randint(1, 61), random.randint(1, 30)] for _ in range(num_forces)])
    #dirs = np.array([[0,1], [0,1], [0,1]])
    #positions = np.array([[61,1], [1,1], [30, 1]])
    nodes, mats, els, loads = beam_rand(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions)
    print(loads)
    '''

    # Initialize the design variables
    change = 10 # Change in the design variable
    g = 0 # Constraint
    rho = volfrac * np.ones(ny*nx, dtype=float) # Initialize the density
    sensi_rho = np.ones(ny*nx) # Initialize the sensitivity
    rho_old = rho.copy() # Initialize the density history
    d_c = np.ones(ny*nx) # Initialize the design change

    r_min = np.linalg.norm(nodes[0,1:3] - nodes[1,1:3]) * 4 # Radius for the sensitivity filter
    centers = center_els(nodes, els) # Calculate centers
    E = mats[0,0] # Young modulus
    nu = mats[0,1] # Poisson ratio
    k = np.array([1/2-nu/6,1/8+nu/8,-1/4-nu/12,-1/8+3*nu/8,-1/4+nu/12,-1/8-nu/8,nu/6,1/8-3*nu/8]) # Coefficients
    kloc = E/(1-nu**2)*np.array([ [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]], 
    [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
    [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
    [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
    [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
    [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
    [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
    [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]]]); # Local stiffness matrix
    assem_op, bc_array, neq = ass.DME(nodes[:, -2:], els, ndof_el_max=8) 

    iter = 0
    for _ in range(niter):
        iter += 1

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

        compliance = rhs_vec.T.dot(disp)

        # Sensitivity analysis
        sensi_rho[:] = (np.dot(UC[els[:,-4:]].reshape(nx*ny,8),kloc) * UC[els[:,-4:]].reshape(nx*ny,8) ).sum(1)
        d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
        d_c[:] = density_filter(centers, r_min, rho, d_c)

        # Optimality criteria
        rho_old[:] = rho
        rho[:], g = optimality_criteria(nx, ny, rho, d_c, g)

        # Compute the change
        change = np.linalg.norm(rho.reshape(nx*ny,1)-rho_old.reshape(nx*ny,1),np.inf)

    return rho

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