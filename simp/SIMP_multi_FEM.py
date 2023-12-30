from os import path, makedirs
import random
import multiprocessing
import logging
import time
import numpy as np
from scipy.sparse.linalg import spsolve
import solidspy.assemutil as ass # Solidspy 1.1.0
from Utils.beams import *
from Utils.SIMP_utils import *

np.seterr(divide='ignore', invalid='ignore')

# Initialize variables
length = 10
height = 10
nx = 60
ny= 60
niter = 100
penal = 3
Emin=1e-9
Emax=1.0

# Optimise function
def optimise(dirs, positions, load_x, load_y, bc):
    nodes, mats, els, loads = beam(L=length, H=height, nx=nx, ny=ny, dirs=dirs, positions=positions)

    # Initialize the design variables
    change = 10 # Change in the design variable
    g = 0 # Constraint
    rho = 0.5 * np.ones(ny*nx, dtype=float) # Initialize the density
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
    # System assembly
    stiff_mat = sparse_assem(els, mats, neq, assem_op, kloc)
    rhs_vec = ass.loadasem(loads, bc_array, neq)

    # System solution
    disp = spsolve(stiff_mat, rhs_vec)
    UC = pos.complete_disp(bc_array, nodes, disp)
    _, stress_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)

    UC_x = UC[:,0]
    UC_y = UC[:,1]
    stress_x = stress_nodes[:,0]
    stress_y = stress_nodes[:,1]
    stress_xy = stress_nodes[:,2]

    for _ in range(niter):
        iter += 1

        # Check convergence
        if change < 0.01:
            print('Convergence reached')
            break

        # Change density 
        mats[:,2] = Emin+rho**penal*(Emax-Emin)

        # System assembly
        stiff_mat = sparse_assem(els, mats, neq, assem_op, kloc)
        rhs_vec = ass.loadasem(loads, bc_array, neq)

        # System solution
        disp = spsolve(stiff_mat, rhs_vec)
        UC = pos.complete_disp(bc_array, nodes, disp)
        _, stress_nodes = pos.strain_nodes(nodes, els, mats[:,:2], UC)

        # Sensitivity analysis
        sensi_rho[:] = (np.dot(UC[els[:,-4:]].reshape(nx*ny,8),kloc) * UC[els[:,-4:]].reshape(nx*ny,8) ).sum(1)
        d_c[:] = (-penal*rho**(penal-1)*(Emax-Emin))*sensi_rho
        d_c[:] = density_filter(centers, r_min, rho, d_c)

        # Optimality criteria
        rho_old[:] = rho
        rho[:], g = optimality_criteria(nx, ny, rho, d_c, g)

        # Compute the change
        change = np.linalg.norm(rho.reshape(nx*ny,1)-rho_old.reshape(nx*ny,1),np.inf)

    return bc.flatten(), load_x.flatten(), load_y.flatten(), UC_x, UC_y, stress_x, stress_y, stress_xy, rho

if __name__ == "__main__":
    start_time = time.perf_counter()

    # Initialize storage variables
    input_bc = []
    input_load = []
    output_rho = []
    tasks = []
    results = []

    # Create tasks
    a = True
    iter = 0

    directions = [[0,1], [0,-1]]
    vols = [0.5,0.6,0.7]
    for _ in range(5):
        volfrac = random.choice(vols)
        num_forces = random.randint(1,2)
        dirs = np.array([random.choice(directions) for _ in range(num_forces)])
        positions = np.array([[random.randint(1, 61), random.randint(1, 30)] for _ in range(num_forces)])

        #dirs = np.array([[0,-1], [0,1], [1,0]])
        #positions = np.array([[61,30], [1,30], [30, 1]])

        load_x = np.zeros((nx+1, ny+1), dtype=int)
        load_y = np.zeros((nx+1, ny+1), dtype=int)
        load_x[-positions[:,0], -positions[:,1]] = dirs[:,0]
        load_y[-positions[:,0], -positions[:,1]] = dirs[:,1]

        bc = np.zeros((nx + 1, ny + 1))
        bc[:, 0] = 1

        iter += 1
        task = (dirs, positions, load_x, load_y, bc)
        tasks.append(task)
    
    # Create pool
    num_processor = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=num_processor)
    print(num_processor)

    results = pool.starmap(optimise, tasks)
    pool.close()
    pool.join()

    final_input_bc = []
    final_input_load_x = []
    final_input_load_y = []
    final_input_uc_x = []
    final_input_uc_y = []
    final_input_stress_x = []
    final_input_stress_y = []
    final_input_stress_xy = []
    final_output_rho = []

    # Unpack results
    for result in results:
        final_input_bc.append(result[0])
        final_input_load_x.append(result[1])
        final_input_load_y.append(result[2])
        final_input_uc_x.append(result[3])
        final_input_uc_y.append(result[4])
        final_input_stress_x.append(result[5])
        final_input_stress_y.append(result[6])
        final_input_stress_xy.append(result[7])
        final_output_rho.append(result[8])

    # Save data
    dir = './results'
    if not path.exists(dir): makedirs(dir)
    np.savetxt(dir + '/bc.txt', np.array(final_input_bc), fmt="%.3f")
    np.savetxt(dir + '/load_x.txt', np.array(final_input_load_x), fmt="%.3f")
    np.savetxt(dir + '/load_y.txt', np.array(final_input_load_y), fmt="%.3f")
    np.savetxt(dir + '/uc_x.txt', np.array(final_input_uc_x), fmt="%.3f")
    np.savetxt(dir + '/uc_y.txt', np.array(final_input_uc_y), fmt="%.3f")
    np.savetxt(dir + '/stress_x.txt', np.array(final_input_stress_x), fmt="%.3f")
    np.savetxt(dir + '/stress_y.txt', np.array(final_input_stress_y), fmt="%.3f")
    np.savetxt(dir + '/stress_xy.txt', np.array(final_input_stress_xy), fmt="%.3f")
    np.savetxt(dir + '/output.txt', np.array(final_output_rho), fmt="%.3f")

    # Log time
    end_time = time.perf_counter()
    execution_time = end_time - start_time

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logsimp.txt')
    logging.info(f"Execution time: {execution_time} seconds in SIMP multiprocessing with {num_processor} processors. [{iter*5} samples]")