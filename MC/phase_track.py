import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
from cemc.tools.phase_boundary_tracker import PhaseBoundaryTracker
from cemc.tools import save_phase_boundary
import pickle as pck
from matplotlib import pyplot as plt
from ase.visualize import view
import json
import numpy as np
from mpi4py import MPI
import gc

eci_vib = {
    "c1_0":0.43,
    "c2_1000_00":-0.045
}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
def main():
    mu_al = -1.06
    mu_al3mg = -1.067
    gs_al_file = "data/bc_10x10x10_Al.pkl"
    gs_al3mg_file = "data/bc_10x10x10_Al3Mg.pkl"
    with open( gs_al_file, 'rb' ) as infile:
        bc_al,cf_al,eci_al = pck.load( infile )
    with open( gs_al3mg_file, 'rb' ) as infile:
        bc_al3mg, cf_al3mg, eci_al3mg = pck.load( infile )

    bc_al.db_name = "temp_al_phase_track.db"
    bc_al3mg.db_name = "temp_al3mg_phase_track.db"
    print (bc_al.db_name)
    print(bc_al3mg.db_name)
    bc_al._store_data()
    bc_al3mg._store_data()

    gs_al = {
        "bc":bc_al,
        "eci":eci_al,
        "cf":cf_al
    }
    #gs_al["bc"].reconfigure_settings()

    gs_al3mg = {
        "bc":bc_al3mg,
        "eci":eci_al3mg,
        "cf":cf_al3mg
    }

    #view(bc_al.atoms)
    #view(bc_al3mg.atoms)
    #exit()
    gs = [gs_al, gs_al3mg]
    boundary_tracker = PhaseBoundaryTracker( gs )

    # Construct common tangent construction
    #T = [200,250,300,310,320,330,340,350,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380]
    mc_args = {
        "mode":"prec",
        "prec":1E-3,
        "equil":True,
        "steps":500000
    }
    res = boundary_tracker.separation_line_adaptive_euler( init_temp=100, stepsize=50, min_step=1.0, mc_args=mc_args, symbols=["Al", "Mg"] )
    print (res)
    if ( rank == 0 ):
        save_phase_boundary("data/phae_boundary_almg.h5", res)
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot( res["temperature"], res["mu"] )
    #plt.show()

if __name__ == "__main__":
    main()
