import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
sys.path.insert(2,"/home/dkleiven/Documents/aseJin")

from cemc.mcmc import NucleationSampler, SGCNucleation, CanonicalNucleationMC
#from cemc.mcmc import TransitionPathRelaxer
from ase.ce import BulkCrystal
from cemc.wanglandau.ce_calculator import get_ce_calc
import json
from matplotlib import pyplot as plt
import h5py as h5
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD

def main(outfname,action):
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
    sampler = NucleationSampler( size_window_width=3, \
    chemical_potential=chem_pot, max_cluster_size=15, \
    merge_strategy="normalize_overlap", mpicomm=comm, max_one_cluster=True )

    T = 700
    mc = SGCNucleation( ceBulk.atoms, T, nucleation_sampler=sampler, \
    network_name="c2_1414_1",  network_element="Mg", symbols=["Al","Mg"], \
    chem_pot=chem_pot, allow_solutes=True )

    mg_conc = 0.04
    concentration = {
        "Mg":mg_conc,
        "Al":1.0-mg_conc
    }

    mc_canonical = CanonicalNucleationMC( ceBulk.atoms, T, nucleation_sampler=sampler,
    network_name="c2_1414_1",  network_element="Mg", concentration=concentration )

    if ( action == "barrier" ):
        mc.run(nsteps=100000)
        sampler.save(fname=outfname)
    elif ( action == "barrier_canonical" ):
        mc_canonical.run( nsteps=50000 )
        sampler.save(fname=outfname)
    elif ( action == "trans_path" ):
        mc.find_transition_path( initial_cluster_size=50, max_size_reactant=10, min_size_product=100, folder="data", path_length=50, max_attempts=10 )
        plt.show()
    elif ( action == "relax_path" ):
        relaxer = TransitionPathRelaxer( nuc_mc=mc )
        relaxer.relax_path( initial_path=outfname, n_shooting_moves=1000 )
        relaxer.path2trajectory(fname="data/relaxed_path.traj")
    elif( action == "generate_paths" ):
        relaxer = TransitionPathRelaxer( nuc_mc=mc )
        relaxer.generate_paths( initial_path=outfname, n_paths=10, outfile="data/tse_ensemble.json" )
    elif ( action == "view_tps_indicators" ):
        relaxer = TransitionPathRelaxer( nuc_mc=mc )
        relaxer.plot_path_statistics( path_file=outfname )
        plt.show()

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
    option = sys.argv[1]
    if ( option == "run" ):
        fname = sys.argv[2]
        main(fname,"barrier")
    elif ( option == "barrier_canonical" ):
        fname = sys.argv[2]
        main(fname,"barrier_canonical")
    elif ( option == "plot" ):
        fname = sys.argv[2]
        plot(fname)
    elif ( option == "trans_path" ):
        main(None,"trans_path")
    elif ( option == "relax_path" ):
        fname = sys.argv[2]
        main(fname,"relax_path")
    elif ( option == "generate_paths" ):
        fname = sys.argv[2]
        main(fname,"generate_paths")
    elif ( option == "view_tps_indicators" ):
        fname = sys.argv[2]
        main(fname,"view_tps_indicators")
