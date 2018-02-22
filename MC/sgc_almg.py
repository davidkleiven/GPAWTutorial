import sys
from cemc.mcmc import sgc_montecarlo as sgc
import json
from ase.ce.settings import BulkCrystal
from cemc.wanglandau.ce_calculator import CE
import numpy as np
from mpi4py import MPI
from ase.visualize import view

OUTFILE = "data/almg_15x15x15.json"
db_name = "data/ce_hydrostatic.db"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
def run( mu, temps, save=False ):
    chem_pots = {
        "c1_1":mu
    }

    conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
    ceBulk = BulkCrystal( "fcc", 4.05, None, [21,21,21], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, max_cluster_dia=1.414*4.05, reconf_db=True )

    eci_file = "/home/davidkl/Documents/GPAWTutorial/CE/data/almg_eci.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )

    calc = CE( ceBulk, ecis )
    ceBulk.atoms.set_calculator( calc )

    n_burn = 40000
    n_sample = 100000
    thermo = []
    size = comm.Get_size()
    n_per_proc = int( len(temps)/size )
    if ( rank == comm.Get_size()-1 ):
        my_temps = temps[n_per_proc*rank:]
    else:
        my_temps = temps[n_per_proc*rank:n_per_proc*(rank+1)]

    for T in my_temps:
        print ("{}: Current temperature {}".format(rank, T) )
        mc = sgc.SGCMonteCarlo( ceBulk.atoms, T, symbols=["Al","Mg"] )
        mc.runMC( steps=n_burn, chem_potential=chem_pots )
        mc.runMC( steps=n_sample, chem_potential=chem_pots )
        thermo_properties = mc.get_thermodynamic()
        thermo.append( thermo_properties )

    all_thermos = []
    all_thermos = comm.gather( thermo, root=0 )

    if ( (rank == 0) and save ):
        thermo = []
        for sublist in all_thermos:
            thermo += sublist
        name = "mu%d"%(int(abs(mu)*10000))
        if ( mu < 0.0 ):
            name += "m"
        data_written_to_file = False
        data = {}
        try:
            with open( OUTFILE, 'r' ) as infile:
                data = json.load( infile )

            if ( name in data.keys() ):
                for i,entry in enumerate(thermo):
                    data[name]["singlets"].append( entry["singlets"][0] )
                    data[name]["energy"].append( entry["energy"] )
                    data[name]["heat_capacity"].append( entry["heat_capacity"] )
                    data[name]["temperature"].append( entry["temperature"] )
                    data_written_to_file = True
        except Exception as exc:
            print (str(exc))
            print ("Could not load file! Creating a new one!")

        if ( not data_written_to_file ):
            data[name] = {}
            data[name]["singlets"] = [entry["singlets"][0] for entry in thermo]
            data[name]["temperature"] = list(temps)
            data[name]["energy"] = [entry["energy"] for entry in thermo]
            data[name]["heat_capacity"] = [entry["heat_capacity"] for entry in thermo]
            data[name]["mu"] = mu
            with open( OUTFILE, 'w' ) as outfile:
                json.dump( data, outfile, sort_keys=True, indent=2, separators=(",",":") )
            print ( "Data written to {}".format(OUTFILE) )

    if ( comm.Get_size() == 1 ):
        print (ceBulk.atoms.get_chemical_formula() )
        view( ceBulk.atoms )
def main( argv ):
    maxT = None
    minT = None
    stepsT = None
    save = False
    for arg in argv:
        if ( arg.find("--mu=") != -1 ):
            mu = float( arg.split("--mu=")[-1] )
        elif ( arg.find("--T=") != -1 ):
            temps_str = arg.split("--T=")[-1]
            temps = temps_str.split(",")
            temps = [float(T) for T in temps]
        elif( arg.find("--maxT=") != -1 ):
            maxT = float( arg.split("--maxT=")[-1] )
        elif( arg.find("--minT=") != -1 ):
            minT = float( arg.split("--minT=")[-1] )
        elif ( arg.find("--nT=") != -1 ):
            stepsT = float( arg.split("--nT=")[-1] )
        elif ( arg.find("--save") != -1 ):
            save = True
        else:
            print ( "Unknown key!")

    if ( maxT is not None and minT is not None and stepsT is not None ):
        temps = list( np.linspace( minT, maxT, stepsT )[::-1] )

    run( mu, temps, save=save )

if __name__ == "__main__":
    main( sys.argv[1:] )
