import sys
from phasefield_cxx import PyCahnHilliard
from phasefield_cxx import PyCahnHilliardPhaseField
import numpy as np
import json

FNAME = "data/pseudo_binary_free/adaptive_bias500K_-650mev_cahn_hilliard.json"

def main(conc):
    prefix = "data/almgsi_ch500K/chgl_{}_".format(int(100*conc))
    dim = 2
    L = 1024
    num_gl_fields = 0
    M = 0.1
    alpha = 5.0
    dt = 0.5
    gl_damping = M
    gradient_coeff = []

    with open(FNAME, 'r') as infile:
        info = json.load(infile)
    
    alpha = info["alpha"]
    cahn_free = PyCahnHilliard(info["poly"], bounds=[-0.05, 1.05], penalty=0.0)
    sim = PyCahnHilliardPhaseField(2, L, prefix, cahn_free, M, dt, alpha)

    sim.from_npy_array(random_init(L, conc))
    sim.set_adaptive(1E-10, 0.05)
    sim.build2D()
    #sim.from_file(prefix + "00000100000.grid")

    sim.run(50000, 1000, start=0)
    sim.save_free_energy_map(prefix+"_free_energy_map.grid")


def random_init(L, conc):
    data = np.random.random((L, L))
    data *= 2*conc
    print(np.mean(data))
    return data

if __name__ == '__main__':
    conc = float(sys.argv[1])
    main(conc)
