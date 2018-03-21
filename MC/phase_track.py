from cemc.tools.phase_boundary_tracker import PhaseBoundaryTracker
import pickle as pck
from matplotlib import pyplot as plt
from ase.visualize import view
import json
import numpy as np

eci_vib = {
    "c1_0":0.43,
    "c2_1000_00":-0.045
}
def main():
    mu_al = -1.06
    mu_al3mg = -1.067
    gs_al_file = "data/bc_10x10x10_gsAl.pkl"
    gs_al3mg_file = "data/bc_10x10x10_gsAl3Mg.pkl"
    with open( gs_al_file, 'rb' ) as infile:
        bc_al,cf_al,eci_al = pck.load( infile )
    with open( gs_al3mg_file, 'rb' ) as infile:
        bc_al3mg, cf_al3mg, eci_al3mg = pck.load( infile )

    gs_al = {
        "bc":bc_al,
        "eci":eci_al,
        "cf":cf_al,
        "linvib":lvc.LinearVibCorrection(eci_vib)
    }

    gs_al3mg = {
        "bc":bc_al3mg,
        "eci":eci_al3mg,
        "cf":cf_al3mg,
        "linvib":lvc.LinearVibCorrection(eci_vib)
    }
    #view(bc_al.atoms)
    #view(bc_al3mg.atoms)
    #exit()
    boundary_tracker = PhaseBoundaryTracker( gs_al, gs_al3mg )
    zero_kelvin_separation = boundary_tracker.get_zero_temperature_mu_boundary()
    print ("0K phase boundary {}".format(zero_kelvin_separation) )

    # Construct common tangent construction
    #T = [200,250,300,310,320,330,340,350,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380]
    T = [200,250,300,350,400,450,500]
    res = boundary_tracker.separation_line( np.array(T) )
    with open("data/phase_boundary.json",'w') as outfile:
        json.dump( res, outfile, sort_keys=True, indent=2, separators=(",",":") )
    print (res["msg"])
    print (bc_al.atoms.get_chemical_formula())
    print (bc_al3mg.atoms.get_chemical_formula())
    view(bc_al.atoms)
    view(bc_al3mg.atoms)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( res["temperature"], res["mu"] )
    plt.show()
if __name__ == "__main__":
    main()
