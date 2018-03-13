import json
import matplotlib as mpl
mpl.rcParams["font.size"] = 18
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from ase.units import kJ,mol

special_compounds = {
    "Al12Mg17(E)":(-145.68/58 + 3.74*12.0/29.0 + 1.59*17.0/29.0,0.586)
}
def main():
    show_ids = False
    show_special_compounds = True
    fname = "data/almg_formation_energy.json"
    with open(fname,'r') as infile:
        data = json.load( infile )

    ce_form = np.array( data["ce_formation"] )*mol/kJ
    mg_conc = data["mg_conc"]
    dft_form = np.array( data["dft_formation"] )*mol/kJ
    dft_mg_conc = data["dft_mg_conc"]
    ids = data["dft_ids"]
    #dft_mg_conc = mg_conc
    points = np.vstack( (dft_mg_conc,dft_form) ).T
    qhull = ConvexHull(points)

    # Sort mg_conc
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( dft_mg_conc, dft_form, "o", mfc="none", label="DFT")
    ax.plot( mg_conc, ce_form, "x", label="CE" )
    ax.legend( frameon=False )

    # Plot convex hull
    for simplex in qhull.simplices:
        if ( points[simplex,1][0] > 0.0 or points[simplex,1][1] > 0.0 ):
            continue
        ax.plot( points[simplex,0], points[simplex,1], color="#fb8072" )

    if ( show_ids ):
        for i in range(len(dft_mg_conc) ):
            ax.text( dft_mg_conc[i], dft_form[i], "{}".format(ids[i]), fontsize=8 )

    print (kJ,mol)
    if ( show_special_compounds ):
        color= "#4daf4a"
        for key,value in special_compounds.iteritems():
            print (key,value)
            ax.plot( value[1], value[0]*mol/kJ, "x", color=color )
            ax.text( value[1], value[0]*mol/kJ, "{}".format(key), fontsize=12 )

    ax.set_xlabel( "Mg concentration" )
    ax.set_ylabel( "Enthalpy of formation (kJ/mol)" )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axhline(0.0, color="#80b1d3", ls="--")
    plt.show()

if __name__ == "__main__":
    main()
