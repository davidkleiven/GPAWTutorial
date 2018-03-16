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
from ase.units import kB, kJ,mol
from scipy import integrate
from cemc.tools import free_energy as fe
from cemc.mfa.mean_field_approx import MeanFieldApprox
import pickle as pck
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.visualize import view

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
    try:
        chem_pot = []
        T = []
        conc = []
        min_mu = np.inf
        max_mu = -np.inf
        for key,value in data.iteritems():
            if ( value["mu_c1_0"] < min_mu ):
                min_mu = value["mu_c1_0"]

            if ( value["mu_c1_0"] > max_mu ):
                max_mu = value["mu_c1_0"]

        Z = [[0,0],[0,0]]

        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        all_mapped_mu = []
        all_max_T = []
        all_min_T = []
        for key,value in data.iteritems():
            value = sort_based_on_temp(value)
            chem_pot.append( value["mu_c1_0"] )
            mapped_mu = (value["mu_c1_0"]-min_mu)/(max_mu-min_mu)
            all_mapped_mu.append( mapped_mu )
            T = value["temperature"]
            all_max_T.append( np.max(T) )
            all_min_T.append( np.min(T) )
            conc = value["singlet_c1_0"]
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
    except Exception as exc:
        print (str(exc))
        return None

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

def free_energy( data, bc, eci_gs ):
    """
    Computes the free energy by integrating along curves of constant chemical potential
    """
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    eng = fe.FreeEnergy()
    for key,value in data.iteritems():
        value = sort_based_on_temp(value)
        T = np.array( value["temperature"] )
        U = np.array( value["energy"] )
        #mf = MeanFieldApprox(bc,symbols=["Al","Mg"])
        #eng_low_temp = fe.FreeEnergy( limit="lte", mfa=mf )
        mu = {"c1_0":value["mu_c1_0"][0]}
        sng = {"c1_0":value["singlet_c1_0"]}
        sgc_E = eng.get_sgc_energy( U/len(bc.atoms), sng, mu )
        #sgc_E_low = eng_low_temp.get_sgc_energy( U, sng, mu )
        #eng_low_temp.chemical_potential = mu # The low temperature expansion requires mu
        res = eng.free_energy_isochemical( T=T, sgc_energy=sgc_E, nelem=2 )
        #res_low = eng_low_temp.free_energy_isochemical( T=T, sgc_energy=sgc_E/1000.0, nelem=2 )
        #beta_mf = np.linspace(1.0/(kB*T[0]), 1.0/(kB*T[-1]),100.0)
        #mf_energy = mf.free_energy( beta_mf, chem_pot=mu )
        G = np.array( res["free_energy"] )*mol/kJ
        ax.plot( res["temperature"], G, marker="o", label="{}".format(value["mu_c1_0"][0]))
        #ax.plot( res_low["temperature"], res_low["free_energy"], marker="x", label="LTE" )
        #T_mf = 1.0/(kB*beta_mf)
        #ax.plot( T_mf, mf_energy, label="MFA" )
    ax.legend( loc="best", frameon=False )
    ax.set_xlabel( "Temperature (K)" )
    ax.set_ylabel( "Free energy (kJ/mol)" )
    return fig

def main( argv ):
    fname = argv[0]
    with open( fname, 'r' ) as infile:
        data = json.load(infile)

    pickle_name = fname.split(".")[0]+".pkl"
    pickle_name = "data/bc_10x10x10_linvib.pkl"
    with open( pickle_name, 'rb' ) as infile:
        bc,cf,eci = pck.load( infile )

    # Plot heat capacities
    gr_spec = {"hspace":0.0}
    fig, ax = plt.subplots(nrows=2, sharex=True, gridspec_kw=gr_spec)
    colors = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6']
    counter = 0
    mu = []
    max_T = []
    max_conc = []
    for key,value in data.iteritems():
        value = sort_based_on_temp(value)
        color = colors[counter%len(colors)]
        mu.append( value["mu_c1_0"] )
        U = np.array(value["energy"])*mol/(len(bc.atoms)*kJ)
        energy_interp = UnivariateSpline( value["temperature"], U, k=3, s=1 )
        ax[0].plot( value["temperature"], U, "o", mfc="none", color=color )
        T = np.linspace( np.min(value["temperature"]), np.max(value["temperature"]), 500)
        ax[0].plot( T, energy_interp(T), color=color )

        singl = np.array( value["singlet_c1_0"] )
        x = 0.5*(1.0+singl)
        ax[1].plot( value["temperature"], x, label="{}".format(value["mu_c1_0"]), color=color, marker="o",mfc="none")
        counter += 1
        #indx_max = np.argmax( Cv(T) )
        #max_T.append( T[indx_max] )
        #interp_conc = UnivariateSpline( value["temperature"], x, k=3, s=1 )
        #x_interp = interp_conc(T)
        #max_conc.append( x_interp[indx_max] )


    fig.subplots_adjust(wspace=0)
    ax[1].set_xlabel( "Temperature (K)" )
    ax[0].set_ylabel( "Internal energy (kJ/mol)")
    ax[1].set_ylabel( "Al conc.")
    ax[1].legend( frameon=False )
    isochemical_potential( data, max_T, max_conc )
    #mu_T_phase_diag( mu, max_T, 200.0, 900.0 )
    print (cf)
    calc = ClusterExpansion( bc, cluster_name_eci=eci, init_cf=cf, logfile=None )
    bc.atoms.set_calculator(calc)
    view(bc.atoms)
    free_energy(data,bc,eci)
    plt.show()

if __name__ == "__main__":
    main( sys.argv[1:] )
