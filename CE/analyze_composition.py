import json
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt

def cluster_per_atom( data ):
    cluster_names = ["c2_1000_1_1","c2_1225_1_1"]
    mg_atoms = np.array( data["n_mg"])
    clusters = []
    for name in cluster_names:
        clusters.append( data[name] )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( mg_atoms/64.0, clusters[0]/mg_atoms, "-o", label="N.N" )
    ax.plot( mg_atoms/64.0, clusters[1]/mg_atoms, "-o", label="S.N.N" )
    ax.legend( loc="best", frameon=False )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Mg concentration")
    ax.set_ylabel("Number of clusters per atom")
    plt.show()

def main():
    fname = "composition_sweep.json"
    with open(fname,'r') as f:
        data = json.load(f)

    cluster_per_atom( data )

if __name__ == "__main__":
    main()
