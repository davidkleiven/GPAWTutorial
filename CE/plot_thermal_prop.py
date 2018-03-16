import json
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import numpy as np
fname = "data/thermal_prop_almg_fcc.json"
def main():
    with open(fname,'r') as infile:
        data = json.load( infile )

    mg_conc = data["mg_conc"]
    srt_indx = np.argsort(mg_conc)
    mg_conc = [mg_conc[indx] for indx in srt_indx]
    B = [data["bulk_mod"][indx] for indx in srt_indx]
    T_D = [data["debye_temp"][indx] for indx in srt_indx]
    fig, ax = plt.subplots( nrows=2, gridspec_kw={"hspace":0.0},sharex=True )
    ax[0].plot( mg_conc, B, marker="o" )
    ax[1].plot( mg_conc, T_D, marker="o" )
    ax[1].set_xlabel( "Mg concentration" )
    ax[1].set_ylabel( "Debye Temp. (K)" )
    ax[0].set_ylabel( "Bulk mod. (GPa)" )
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)
    #ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
