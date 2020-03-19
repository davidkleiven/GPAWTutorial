import sys
from apal_cxx import PyCahnHilliard
from apal_cxx import PyCahnHilliardPhaseField
import numpy as np
import json

FNAME = "data/pseudo_binary_free/adaptive_bias500K_-650mev_cahn_hilliard.json"

def main(conc):
    prefix = "/work/sophus/cahn_hilliard_phase_separation3D/ch_{}_".format(int(100*conc))
    dim = 3
    dx = 10.0  # Step size in angstrom
    L = 128
    num_gl_fields = 0
    M = 0.1
    alpha = 5.0
    dt = 0.5
    gl_damping = M
    gradient_coeff = []

    with open(FNAME, 'r') as infile:
        info = json.load(infile)
    
    alpha = info["alpha"]/dx**2
    cahn_free = PyCahnHilliard(info["poly"], bounds=[-0.05, 1.05], penalty=0.0)
    sim = PyCahnHilliardPhaseField(3, L, prefix, cahn_free, M, dt, alpha)

    #sim.from_npy_array(random_init(L, conc))
    sim.set_adaptive(1E-10, 0.05)
    #sim.build3D()
    sim.from_file(prefix + "00000160000.grid")

    sim.run(100000, 10000, start=160000)
    #sim.save_free_energy_map(prefix+"_free_energy_map.grid")


def random_init(L, conc):
    data = np.random.random((L, L, L))
    data *= 2*conc
    print(np.mean(data))
    return data


def random_normal_init(L, conc):
    from scipy.stats import beta
    a = 1
    b = a/conc - 1

    data = beta.rvs(a, b, size=(L, L))
    print(np.mean(data), np.min(data), np.max(data))
    return data


if __name__ == '__main__':
    conc = float(sys.argv[1])
    main(conc)
