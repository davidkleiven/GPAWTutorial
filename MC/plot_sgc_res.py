import sys
import json
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
from scipy.interpolate import UnivariateSpline
from matplotlib import cm

def sort_based_on_temp( entry ):
    T = entry["temperature"]
    srt_indx = np.argsort(T)
    for key in entry.keys():
        try:
            entry[key] = [entry[key][indx] for indx in srt_indx]
        except:
            pass
    return entry

def isochemical_potential( data,max_T, max_conc ):
    """
    Creates a color plot with over composition chemical potential
    """
    chem_pot = []
    T = []
    conc = []
    min_mu = np.inf
    max_mu = -np.inf
    for key,value in data.iteritems():
        if ( value["mu"] < min_mu ):
            min_mu = value["mu"]

        if ( value["mu"] > max_mu ):
            max_mu = value["mu"]

    Z = [[0,0],[0,0]]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    all_mapped_mu = []
    all_max_T = []
    all_min_T = []
    for key,value in data.iteritems():
        value = sort_based_on_temp(value)
        chem_pot.append( value["mu"] )
        mapped_mu = (value["mu"]-min_mu)/(max_mu-min_mu)
        all_mapped_mu.append( mapped_mu )
        T = value["temperature"]
        all_max_T.append( np.max(T) )
        all_min_T.append( np.min(T) )
        conc = value["singlets"]
        conc = (np.array(conc)+1.0)/2.0
        conc = 1.0-conc
        ax.plot( conc, T, color=cm.nipy_spectral(mapped_mu), lw=5 )

    max_conc = 1.0-np.array( max_conc )
    srt_indx = np.argsort( max_conc )
    max_conc = [max_conc[indx] for indx in srt_indx]
    max_T = [max_T[indx] for indx in srt_indx]

    filterered_max_conc = []
    filtered_max_T = []
    for i in range(len(max_T)):
        if (( max_T[i] >= max(all_max_T) or ( max_T[i] <= min(all_min_T) ))):
            continue
        filtered_max_T.append( max_T[i] )
        filterered_max_conc.append( max_conc[i] )
    max_conc = filterered_max_conc
    max_T = filtered_max_T
    ax.plot( max_conc, max_T, lw=2, color="#de2d26", marker="o" )
    chem_pot.sort()
    fig_info = plt.figure()
    ax_info = fig_info.add_subplot(1,1,1)
    Cb_info = ax_info.contourf(Z,chem_pot, cmap="nipy_spectral")
    cbar = fig.colorbar(Cb_info)
    cbar.set_label( "Chemical potential (eV/atom)")
    ax.set_xlabel( "Mg concentration" )
    ax.set_ylabel( "Temperature (K)")
    return fig

def mu_T_phase_diag( max_mu, max_T, Tmin, Tmax ):
    filtered_mu = []
    filtered_max_T = []
    for i in range(len(max_T)):
        if ( (max_T[i] >= Tmax-1.0) or (max_T[i] <= Tmin+1.0) ):
            continue
        filtered_mu.append( max_mu[i] )
        filtered_max_T.append( max_T[i] )

    srt_indx = np.argsort(filtered_mu)
    filtered_mu = [filtered_mu[indx] for indx in srt_indx]
    filtered_max_T = [filtered_max_T[indx] for indx in srt_indx]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( filtered_mu, filtered_max_T, marker="o" )
    ax.set_xlabel( "Chemical potential (eV/atom)" )
    ax.set_ylabel( "Temperature (K)" )
    return fig

def main( argv ):
    fname = argv[0]
    with open( fname, 'r' ) as infile:
        data = json.load(infile)

    # Plot heat capacities
    gr_spec = {"hspace":0.0}
    fig, ax = plt.subplots(nrows=3, sharex=True, gridspec_kw=gr_spec)
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
    counter = 0
    mu = []
    max_T = []
    max_conc = []
    for key,value in data.iteritems():
        value = sort_based_on_temp(value)
        color = colors[counter%len(colors)]
        mu.append( value["mu"] )
        energy_interp = UnivariateSpline( value["temperature"], value["energy"], k=3, s=1 )
        ax[0].plot( value["temperature"], value["energy"], "o", mfc="none", color=color )
        T = np.linspace( np.min(value["temperature"]), np.max(value["temperature"]), 500)
        ax[0].plot( T, energy_interp(T), color=color )
        Cv = energy_interp.derivative()
        ax[1].plot( T, Cv(T), color=color )

        singl = np.array( value["singlets"] )
        x = 0.5*(1.0+singl)
        ax[2].plot( value["temperature"], x, label="{}".format(value["mu"]), color=color, marker="o",mfc="none")
        counter += 1
        indx_max = np.argmax( Cv(T) )
        max_T.append( T[indx_max] )
        interp_conc = UnivariateSpline( value["temperature"], x, k=3, s=1 )
        x_interp = interp_conc(T)
        max_conc.append( x_interp[indx_max] )


    fig.subplots_adjust(wspace=0)
    ax[2].set_xlabel( "Temperature (K)" )
    ax[1].set_ylabel( "Heat capcacity" )
    ax[0].set_ylabel( "Internal energy")
    ax[2].set_ylabel( "Al conc.")
    ax[2].legend( frameon=False )
    isochemical_potential( data, max_T, max_conc )
    mu_T_phase_diag( mu, max_T, 200.0, 900.0 )
    plt.show()

if __name__ == "__main__":
    main( sys.argv[1:] )
