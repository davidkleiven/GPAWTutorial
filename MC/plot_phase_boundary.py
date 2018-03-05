import json
import sys
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
from ase.units import mol,kJ
import numpy as np

def main(argv):
    fname = argv[0]
    with open(fname,'r') as infile:
        data = json.load(infile)

    # T-mu phase diagram
    mu = np.array( data["mu"] )
    mu = mu-mu[0]
    mu *= (1000.0*mol/kJ)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( mu, data["temperature"], marker="x" )
    ax.set_xlabel( "\$\mu - \mu(T=0K)\$ (J/mol)" )
    ax.set_ylabel( "Temperature (K)" )
    ax.spines["right"].set_visible( False )
    ax.spines["top"].set_visible( False )

    fig_comp = plt.figure()
    ax_comp = fig_comp.add_subplot(1,1,1)
    singl1 = np.array( data["singlet1"] )
    singl2 = np.array( data["singlet2"] )
    al_conc1 = (1.0+singl1)/2.0
    al_conc2 = (1.0+singl2)/2.0
    mg_conc1 = 1.0-al_conc1
    mg_conc2 = 1.0-al_conc2

    ax_comp.plot( mg_conc1, data["temperature"], marker="x" )
    ax_comp.plot( mg_conc2, data["temperature"], marker="o", mfc="none" )
    ax_comp.spines["right"].set_visible(False)
    ax_comp.spines["top"].set_visible(False)
    ax_comp.set_xlabel( "Mg concentration" )
    ax_comp.set_ylabel( "Temperature (K)" )
    plt.show()

if __name__ == "__main__":
    main( sys.argv[1:] )
