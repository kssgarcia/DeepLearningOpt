from os import path, makedirs
import topy
import numpy as np
import time
import logging

# Start the timer
start_time = time.time()

input_load = []
input_bc = []
ouput_rho = []

iter = 6
for l in range(3600, 3600-iter, -1):
     load = np.zeros((60, 60), dtype=int)
     load[(l-1)%60, int((l-1)/60)] = 1

     for volfrac in [0.9,0.8,0.7,0.6,0.5]:
          bc = np.ones((60,60)) * volfrac
          bc[:, 0] = 1

          config = {
               'PROB_NAME': 'beam_2d_reci',
               'PROB_TYPE': 'comp',
               'NUM_ELEM_X': 60,
               'NUM_ELEM_Y': 60,
               'NUM_ELEM_Z': 0,
               'VOL_FRAC': volfrac,
               'FILT_RAD': 1.5,
               'P_FAC': 3.0,
               'DOF_PN': 2,
               'ELEM_K': 'Q4',
               'ETA': '0.5',
               'FXTR_NODE_X': range(1, 61),
               'FXTR_NODE_Y': range(1, 61),
               'LOAD_NODE_Y': l,
               'LOAD_VALU_Y': -1,
               'NUM_ITER': 94,
          }

          t = topy.Topology(config)
          t.set_top_params()
          t, params = topy.optimise(t)
          input_load.append(load.flatten())
          input_bc.append(bc.flatten())
          ouput_rho.append(t.flatten())


dir = './results_topy'
if not path.exists(dir):
        makedirs(dir)

np.savetxt(dir + '/load.txt', np.array(input_load), fmt='%s')
np.savetxt(dir + '/bc.txt', input_bc, fmt="%.1f")
np.savetxt(dir + '/output.txt', np.array(ouput_rho), fmt="%.3f")

# End the timer
end_time = time.time()
execution_time = end_time - start_time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename='logtopy.txt')
logging.info("Execution time: {} seconds using topy library. [{} samples]".format(execution_time, iter*5))