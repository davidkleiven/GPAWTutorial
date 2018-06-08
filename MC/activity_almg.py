import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
sys.path.insert(2,"/home/dkleiven/Documents/aseJin")

from cemc.mcmc import ActivitySampler
#from cemc.mcmc import TransitionPathRelaxer
from ase.ce import BulkCrystal
from cemc.wanglandau.ce_calculator import get_ce_calc
import json
from matplotlib import pyplot as plt
import h5py as h5
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

def main():
    conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
    kwargs = {
        "crystalstructure":"fcc", "a":4.05, "size":[4,4,4], "basis_elements":[["Al","Mg"]],
        "conc_args":conc_args, "db_name":"data/temporary_bcnucleationdb.db",
        "max_cluster_size":4
    }
    ceBulk = BulkCrystal( **kwargs )
    print (ceBulk.basis_functions)

    eci_file = "data/ce_hydrostatic.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    print (ecis)
    #calc = CE( ceBulk, ecis, size=(3,3,3) )
    calc = get_ce_calc( ceBulk, kwargs, ecis, size=[10,10,10], free_unused_arrays_BC=True )
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )

    T = 500
    temps = [200,300,400,500,600,700,800]
    mg_concs = [0.005,0.01,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.70,0.75,0.8,0.85,0.9]
    mg_concs = np.linspace(0.005,0.995,100)
    act_coeffs = []
    eff_conc = []
    for T in temps:
        for c_mg in mg_concs:
            comp = {"Mg":c_mg,"Al":1.0-c_mg}
            calc.set_composition(comp)
            act_sampler = ActivitySampler( ceBulk.atoms, T, moves=[("Al","Mg")], mpicomm=comm)
            act_sampler.runMC( mode="fixed", steps=100000 )
            thermo = act_sampler.get_thermodynamic()
            act_sampler.save( fname="data/effective_concentration_full_range.db" )
    #res = ActivitySampler.effective_composition("data/effective_concentration.json")
    #mg_concs = res[500]["conc"]["Mg"]
    #eff = res[500]["eff_conc"]["Mg"]
    #plt.plot( mg_concs, eff, marker="x" )
    #plt.show()


if __name__ == "__main__":
    main()
