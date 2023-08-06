from os import path, makedirs
from .visualisation import *
from .topology import *
from .topy_logging import *


__all__ = ['optimise']

def optimise(topology):
    etas_avg = []
    # Optimising function:
    def _optimise(t):
        t.fea()
        t.sens_analysis()
        t.filter_sens_sigmund()
        t.update_desvars_oc()
        # Below this line we print info and create images or geometry:
        params = {
            'prefix': t.probname,
            'iternum': t.itercount,
            'time': 'none',
            'filetype': 'png',
            'dir': './'
        }

        return t.desvars, params

    # Try CHG_STOP criteria, if not defined (error), use NUM_ITER for iterations:
    try:
        while topology.change > topology.chgstop:
            t_, params_ = _optimise(topology)
    except AttributeError:
        for _ in range(topology.numiter):
            t_, params_ = _optimise(topology)
    
    return t_, params_
