from cemc.tools.phase_boundary_tracker import PhaseBoundaryTracker
import pickle as pck
from matplotlib import pyplot as plt

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
        "cf":cf_al
    }

    gs_al3mg = {
        "bc":bc_al3mg,
        "eci":eci_al3mg,
        "cf":cf_al3mg
    }
    boundary_tracker = PhaseBoundaryTracker( gs_al, gs_al3mg )
    zero_kelvin_separation = boundary_tracker.get_zero_temperature_mu_boundary()
    print ("0K phase boundary {}".format(zero_kelvin_separation) )

    # Construct common tangent construction
    res = boundary_tracker.separation_line( 100.0, 600.0, nsteps=40 )
    print (res["msg"])
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( res["temperature"], res["mu"] )
    plt.show()
if __name__ == "__main__":
    main()
