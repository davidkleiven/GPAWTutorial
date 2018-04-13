import numpy as np
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
from matplotlib import cm

concs = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95]

def enthalpy_form():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    markers = ["o","x","^","v","d"]

    all_data = []
    folder = "data/mc_hcp"
    for conc in concs:
        new_data = {}
        fname = "{}/thermo_{}.csv".format(folder,conc)
        T, conc, H_form = np.loadtxt( fname, delimiter=",", unpack=True, usecols=(0,2,3) )
        new_data["T"] = T[1:]
        new_data["H"] = H_form[1:]
        new_data["conc"] = conc[1:]

        all_data.append(new_data)

    n_temps = len( all_data[0]["T"] )
    Tmax = np.max(all_data[0]["T"] )
    Tmin = np.min(all_data[0]["T"])
    for i in range(n_temps):
        T = all_data[0]["T"][i]
        form = []
        conc = []
        for dset in all_data:
            form.append( dset["H"][i] )
            conc.append( dset["conc"][i] )

        srt_indx = np.argsort(conc)
        conc = [conc[indx] for indx in srt_indx]
        form = [form[indx] for indx in srt_indx]
        conc.append(1.0)
        form.append(0.0)
        mapped_temp = (T-Tmin)/(Tmax-Tmin)

        ax.plot( conc, form, marker=markers[i%len(markers)], color=cm.copper(mapped_temp), label="{}K".format(T) )
    ax.set_xlabel( "Mg concentration" )
    ax.set_ylabel( "Enthalpy of formation (kJ/mol)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    #ax.legend()

    Z = [[0,0],[0,0]]
    fig_info = plt.figure()
    #temp_info = [np.min(T),np.max(T)]
    ax_info = fig_info.add_subplot(1,1,1)
    temp_info = np.linspace(Tmin,Tmax,256)
    Cb_info = ax_info.contourf(Z,temp_info, cmap="copper")

    cbar = fig.colorbar(Cb_info, orientation="horizontal", ticks=[100,300,500,700] )
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.set_xticklabels([100,300,500,700])
    cbar.ax.xaxis.set_label_position("top")

    #cbar.ax.tick_params(axis='x',direction='in',labeltop='on')
    cbar.set_label( "Temperature (K)")

def main():
    enthalpy_form()
    plt.show()

if __name__ == "__main__":
    main()
