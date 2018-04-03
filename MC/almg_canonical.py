import sys
from cemc.mcmc import Montecarlo
from ase.db import connect
from ase.ce import BulkCrystal
import json
from cemc.wanglandau.ce_calculator import get_ce_calc
import numpy as np
from matplotlib import pyplot as plt
from mpi4py import MPI
from cemc.tools import CanonicalFreeEnergy

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
        temps = argv[1].split(",")
        temps = [float(temp) for temp in temps]
        excess(temps)

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
    return result

def excess( temps ):
    ref_energies = {
        "Al":-3.73667187,
        "Mg":-1.59090625
    }
    res = get_free_energies()
    plt.plot( res["Al950Mg50"]["temperature"], res["Al950Mg50"]["free_energy"], "x" )
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for T in temps:
        excess = []
        concs = []
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
            #excess.append( entry["free_energy"][indx] )
        ax.plot( concs, excess, "x", label="{}K".format(int(T)) )
    plt.show()


if __name__ == "__main__":
    main( sys.argv[1:] )
