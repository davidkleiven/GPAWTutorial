import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
from cemc.tools.phase_boundary_tracker import PhaseBoundaryTracker
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
    boundary_tracker = PhaseBoundaryTracker( gs_al, gs_al3mg )
    zero_kelvin_separation = boundary_tracker.get_zero_temperature_mu_boundary()
    print ("0K phase boundary {}".format(zero_kelvin_separation) )

    # Construct common tangent construction
    #T = [200,250,300,310,320,330,340,350,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380]
    mc_args = {
        "mode":"prec",
        "prec":1E-3,
        "equil":True
    }
    res = boundary_tracker.separation_line_adaptive_euler( T0=100, stepsize=50, min_step=1.0, mc_args=mc_args )
    print (res)
    if ( rank == 0 ):
        with open("data/phase_boundary_adaptive_fixed.json",'w') as outfile:
            json.dump( res, outfile, sort_keys=True, indent=2, separators=(",",":") )
        print (res["msg"])
        print (bc_al.atoms.get_chemical_formula())
        print (bc_al3mg.atoms.get_chemical_formula())
    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot( res["temperature"], res["mu"] )
    #plt.show()

if __name__ == "__main__":
    main()
