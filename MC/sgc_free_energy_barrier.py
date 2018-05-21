import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
sys.path.insert(2,"/home/dkleiven/Documents/aseJin")

from cemc.mcmc import SGCFreeEnergyBarrier
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

    chem_pot = {"c1_0":-1.0651526881167124}
    chem_pot = {"c1_0":-1.069}

    T = 300
    mc = SGCFreeEnergyBarrier( ceBulk.atoms, T, symbols=["Al","Mg"], \
    n_windows=20, n_bins=10, min_singlet=0.5, max_singlet=1.0, mpicomm=comm )
    mc.run( nsteps=100000, chem_pot=chem_pot )
    mc.save( fname="data/free_energy_barrier.json" )
    #mc.plot( fname="data/free_energy_barrier.json" )
    #plt.show()


def plot(fname):
    with h5.File(fname,'r') as hf:
        hist = np.array( hf["overall_hist"])

    beta_G = -np.log(hist)
    beta_G -= beta_G[0]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( beta_G, ls="steps" )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel( "Number of Mg atoms" )
    ax.set_ylabel( "\$\\beta \Delta G\$" )
    plt.show()

if __name__ == "__main__":
    main()
