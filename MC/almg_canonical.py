import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
from cemc.mcmc import Montecarlo
from ase.db import connect
from ase.ce import BulkCrystal
import json
from cemc.wanglandau.ce_calculator import get_ce_calc
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI
from cemc.tools import CanonicalFreeEnergy
from ase.units import kJ,mol
from matplotlib import cm
from scipy.interpolate import UnivariateSpline
import json
plt.switch_backend("TkAgg")

mc_db_name = "data/almg_fcc_canonical.db"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run( maxT, minT, n_temp, mg_conc ):
    T = np.linspace(minT,maxT,n_temp)[::-1]
    conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
    kwargs = {
        "crystalstructure":"fcc", "a":4.05, "size":[4,4,4], "basis_elements":[["Al","Mg"]],
        "conc_args":conc_args, "db_name":"data/temporary_bcdb.db",
        "max_cluster_size":4
    }
    ceBulk = BulkCrystal( **kwargs )
    print (ceBulk.basis_functions)

    eci_file = "data/ce_hydrostatic.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    print (ecis)
    #calc = CE( ceBulk, ecis, size=(3,3,3) )
    calc = get_ce_calc( ceBulk, kwargs, ecis, size=[10,10,10] )
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )
    comp = {
        "Al":1.0-mg_conc,
        "Mg":mg_conc
    }
    calc.set_composition(comp)
    print ("Number of atoms: {}".format(len(ceBulk.atoms)) )
    for temp in T:
        print ("Current temperature {}K".format(temp))
        mc_obj = Montecarlo( ceBulk.atoms, temp, mpicomm=comm )
        mode = "prec"
        prec = 1E-5
        mc_obj.runMC( mode=mode, prec=prec )
        thermo = mc_obj.get_thermodynamic()
        thermo["temperature"] = temp
        thermo["prec"] = prec
        thermo["internal_energy"] = thermo.pop("energy")
        thermo["converged"] = True

        if ( rank == 0 ):
            db = connect( mc_db_name )
            db.write( ceBulk.atoms, key_value_pairs=thermo )

def main( argv ):
    option = argv[0]
    if ( option == "del" ):
        formula = argv[1]
        delete( formula )
    elif ( option == "mc" ):
        maxT = float(argv[1])
        minT = float(argv[2])
        n_temp = int(argv[3])
        mg_conc = float(argv[4])
        run( maxT, minT, n_temp, mg_conc )
    elif ( option == "plot" ):
        formula = argv[1]
        plot(formula)
    elif ( option == "excess" ):
        if ( argv[1] != "all" ):
            temps = argv[1].split(",")
            temps = [float(temp) for temp in temps]
        else:
            temps = "all"
        excess(temps)
    elif ( option == "save_free_eng" ):
        free_energies_to_file( "data/free_energies_fcc.json" )

def plot(formula):
    db = connect( mc_db_name )
    energy = []
    temperature = []
    heat_cap = []
    comp = None
    for row in db.select(formula=formula,converged=1):
        energy.append( row.internal_energy )
        temperature.append( row.temperature )
        heat_cap.append( row.heat_capacity )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( temperature, energy )

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot( temperature, heat_cap )
    plt.show()

def delete(formula):
    db = connect( mc_db_name )
    for row in db.select(formula=formula):
        del db[row.id]

def free_energies_to_file(fname):
    free_eng = get_free_energies()
    for key,value in free_eng.iteritems():
        for subkey,subval in value.iteritems():
            if ( isinstance(subval, np.ndarray) ):
                value[subkey] = value[subkey].tolist()

    with open(fname,'w') as outfile:
        json.dump( free_eng, outfile, indent=2, separators=(",",":") )
    print ("Free energies written to {}".format(fname))

def get_free_energies():
    """
    Compute the Free Energy for all entries in the database
    """
    unique_formulas = []
    db = connect( mc_db_name )
    for row in db.select(converged=1):
        if ( row.formula not in unique_formulas ):
            unique_formulas.append(row.formula)

    result = {}
    for formula in unique_formulas:
        conc = None
        internal_energy = []
        temperature = []
        for row in db.select( formula=formula,converged=1):
            if ( conc is None ):
                count = row.count_atoms()
                conc = {}
                for key,value in count.iteritems():
                    conc[key] = float(value)/row.natoms
            internal_energy.append(row.internal_energy/row.natoms) # Normalize by the number of atoms
            temperature.append( row.temperature )
        free_eng = CanonicalFreeEnergy(conc)
        temp,internal,F = free_eng.get( temperature, internal_energy )
        result[formula] = {}
        result[formula]["temperature"] = temp
        result[formula]["internal_energy"] = internal
        result[formula]["free_energy"] = F
        result[formula]["conc"] = conc
        result[formula]["entropy"] = (internal-F)/temp
        result[formula]["TS"] = internal-F
    return result

def find_extremal_points( all_concs, all_excess, show_plot=True ):
    maxima = []
    minima = []
    for conc,excess in zip(all_concs,all_excess):
        maxima_fixed_temp = []
        minima_fixed_temp = []
        spl = UnivariateSpline( conc, excess, k=2, s=0 )
        x = np.linspace( np.min(all_concs), np.max(all_concs), 100 )
        y = spl(x)

        # Brute force find local maximia
        for i in range(1,len(y)-1):
            if ( y[i] > y[i-1] and y[i] > y[i+1] ):
                maxima_fixed_temp.append( x[i] )
            elif ( y[i] < y[i-1] and y[i] < y[i+1] ):
                minima_fixed_temp.append(x[i])
        maxima.append( maxima_fixed_temp )
        minima.append( minima_fixed_temp )
        if ( show_plot ):
            plt.plot(conc,excess)
            plt.plot(x,y)
            plt.show()
    return maxima, minima

def excess( temps ):
    ref_energies = {
        "Al":-3.73667187,
        "Mg":-1.59090625
    }
    if ( temps == "all" ):
        db = connect( mc_db_name )
        all_temps = []
        for row in db.select(converged=1):
            if ( row.temperature not in all_temps ):
                all_temps.append( row.temperature )
        temps = all_temps
    res = get_free_energies()
    plt.plot( res["Al950Mg50"]["temperature"], res["Al950Mg50"]["free_energy"], "x" )
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    markers = ["o","x","^","v","d"]
    colors = ['#b2182b','#ef8a62','#fddbc7','#f7f7f7','#d1e5f0','#67a9cf','#2166ac']
    fig_entropy = plt.figure()
    ax_entropy = fig_entropy.add_subplot(1,1,1)

    fig_mg_weighted = plt.figure()
    ax_mg_weighted = fig_mg_weighted.add_subplot(1,1,1)

    Tmin = np.min(temps)
    Tmax = np.max(temps)
    all_excess = []
    all_concs = []
    all_temps = []
    for count,T in enumerate(temps):
        excess = []
        concs = []
        entropy = []
        temperature = None
        for key,entry in res.iteritems():
            if ( "Mg" in entry["conc"].keys() ):
                mg_conc = entry["conc"]["Mg"]
            else:
                mg_conc = 0.0
            concs.append(mg_conc)
            diff = np.abs(entry["temperature"]-T)
            indx = np.argmin( diff )
            if ( diff[indx] > 1.0 ):
                print ("Warning! Difference {}. Temperature might not be simulated!".format(diff[indx]) )
            excess.append( entry["free_energy"][indx]-ref_energies["Al"]*entry["conc"]["Al"] - ref_energies["Mg"]*entry["conc"]["Mg"] )
            if ( temperature is None ):
                temperature = entry["temperature"][indx]
            excess[-1] += entry["TS"][indx]
            entropy.append( entry["entropy"][indx] )
            #excess.append( entry["free_energy"][indx]+entry["TS"][indx] )
            #excess.append( entry["free_energy"][indx] )
        concs += [0.0,1.0]
        excess += [0.0,0.0]
        entropy += [0.0,0.0]
        srt_indx = np.argsort(concs)
        concs = [concs[indx] for indx in srt_indx]
        excess = np.array( [excess[indx] for indx in srt_indx] )
        entropy = np.array( [entropy[indx] for indx in srt_indx])
        all_excess.append(excess)
        all_concs.append(concs)
        mapped_temp = (temperature-Tmin)/(Tmax-Tmin)
        all_temps.append( temperature )
        ax.plot( concs, excess*mol/kJ, marker=markers[count%len(markers)], label="{}K".format(temperature), mfc="none", color=cm.copper(mapped_temp),lw=2 )
        ax_entropy.plot( concs, 1000.0*entropy*mol/kJ, marker=markers[count%len(markers)], label="{}K".format(temperature), color=cm.copper(mapped_temp),lw=2 , mfc="none")
        ax_mg_weighted.plot( concs, (excess/concs)*mol/kJ,  marker=markers[count%len(markers)], color=cm.copper(mapped_temp),lw=2 , mfc="none" )
    ax.legend(frameon=False)
    ax.set_xlabel( "Mg concentration" )
    ax.set_ylabel( "Enthalpy of formation (kJ/mol)" )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    ax_entropy.set_xlabel( "Mg concentration" )
    ax_entropy.set_ylabel( "Entropy (J/K mol)")
    ax_entropy.spines["right"].set_visible(False)
    ax_entropy.spines["top"].set_visible(False)
    fig_info = plt.figure()
    Z = [[0,0],[0,0]]
    #temp_info = [np.min(T),np.max(T)]
    ax_info = fig_info.add_subplot(1,1,1)
    temp_info = np.linspace(np.min(temps),np.max(temps),256)
    Cb_info = ax_info.contourf(Z,temp_info, cmap="copper")

    cbar = fig.colorbar(Cb_info, orientation="horizontal", ticks=[100,300,500,700] )
    cbar.ax.xaxis.set_ticks_position("top")
    cbar.ax.set_xticklabels([100,300,500,700])
    cbar.ax.xaxis.set_label_position("top")

    #cbar.ax.tick_params(axis='x',direction='in',labeltop='on')
    cbar.set_label( "Temperature (K)")

    cbar2 = fig_entropy.colorbar(Cb_info, orientation="horizontal", ticks=[100,300,500,700] )
    cbar2.ax.xaxis.set_ticks_position("top")
    cbar2.ax.set_xticklabels([100,300,500,700])
    cbar2.ax.xaxis.set_label_position("top")
    cbar2.set_label( "Temperature (K)" )

    all_maxima, all_minima = find_extremal_points( all_concs, all_excess, show_plot=False )

    all_data = []
    for temp,maxima,minima in zip(all_temps,all_maxima,all_minima):
        all_data.append( {"temperature":temp, "maxima":maxima,"minima":minima} )

    with open("data/extremal_points.json", 'w') as outfile:
        json.dump( all_data, outfile, indent=2, separators=(",",":") )
    plt.show()


if __name__ == "__main__":
    main( sys.argv[1:] )
