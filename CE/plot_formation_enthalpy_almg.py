import json
import matplotlib as mpl
mpl.rcParams["font.size"] = 18
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull

def main():
    show_ids = False
    fname = "data/almg_formation_energy.json"
    with open(fname,'r') as infile:
        data = json.load( infile )

    ce_form = data["ce_formation"]
    mg_conc = data["mg_conc"]
    dft_form = data["dft_formation"]
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
    ax.set_xlabel( "Mg concentration" )
    ax.set_ylabel( "Enthalpy of formation (eV/atom)" )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axhline(0.0, color="#80b1d3", ls="--")
    plt.show()

if __name__ == "__main__":
    main()
