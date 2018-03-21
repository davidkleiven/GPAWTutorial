import sys
from cemc.mcmc import sgc_montecarlo as sgc
import json
from ase.ce.settings_bulk import BulkCrystal
from cemc.wanglandau.ce_calculator import CE
import numpy as np
from mpi4py import MPI
from ase.visualize import view
import dill as pck
from cemc.mcmc.linear_vib_correction import LinearVibCorrection

OUTFILE = "data/almg_10x10x10_linvib.json"
pck_file = "data/bc_10x10x10_linvib.pkl"
db_name = "data/ce_hydrostatic.db"
bc_filename = "../CE/data/BC_fcc.pkl"

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

vib_eci = {
"c1_0":0.43,
"c2_1000_1_00":-0.045
}

def run( mu, temps, save=False ):
    chem_pots = {
        "c1_0":mu
    }
    linvib = LinearVibCorrection(vib_eci)

    conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
    #with open(bc_filename,'rb') as infile:
    #    ceBulk = pck.load(infile)
    ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[4,4,4], basis_elements=[["Al","Mg"]], conc_args=conc_args, db_name=db_name,
     max_cluster_size=4 )
    #ceBulk.reconfigure_settings()
    print (ceBulk.basis_functions)

    eci_file = "data/ce_hydrostatic.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    print (ecis)
    calc = CE( ceBulk, ecis, size=[3,3,3] )
    ceBulk.atoms.set_calculator( calc )
    print ("Number of atoms: {}".format(len(ceBulk.atoms)) )
    view(ceBulk.atoms)
    exit()

    n_burn = 40000
    n_sample = 10000
    thermo = []
    #size = comm.Get_size()
    #n_per_proc = int( len(temps)/size )
    #if ( rank == comm.Get_size()-1 ):
    #    my_temps = temps[n_per_proc*rank:]
    #else:
    #    my_temps = temps[n_per_proc*rank:n_per_proc*(rank+1)]

    for T in temps:
        if ( rank == 0 ):
            print ("{}: Current temperature {}".format(rank, T) )
        mc = sgc.SGCMonteCarlo( ceBulk.atoms, T, symbols=["Al","Mg"], mpicomm=comm )
        mc.linear_vib_correction = linvib
        #mc.runMC( steps=n_burn, chem_potential=chem_pots, equil=False )
        equil = {"confidence_level":1E-8}
        mc.runMC( mode="prec", chem_potential=chem_pots, prec=0.01, equil_params=equil )
        if ( rank==0 ):
            print (mc.atoms._calc.eci["c1_0"])
        thermo_properties = mc.get_thermodynamic()
        thermo.append( thermo_properties )
        mc.reset()

    #all_thermos = []
    #all_thermos = comm.gather( thermo, root=0 )

    if ( (rank == 0) and save ):
        #thermo = []
        #for sublist in all_thermos:
        #    thermo += sublist
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
                    for key,value in entry.iteritems():
                        data[name][key].append( value )
                    #data[name]["singlets"].append( entry["singlets"][0] )
                    #data[name]["energy"].append( entry["energy"] )
                    #data[name]["heat_capacity"].append( entry["heat_capacity"] )
                    #data[name]["temperature"].append( entry["temperature"] )
                    data_written_to_file = True
        except Exception as exc:
            print (str(exc))
            print ("Could not load file! Creating a new one!")

        if ( not data_written_to_file ):
            data[name] = {}
            for key in thermo[0].keys():
                for entry in thermo:
                    data[name][key] = [entry[key] for entry in thermo]
            #data[name]["singlets"] = [entry["singlets"][0] for entry in thermo]
            #data[name]["temperature"] = list(temps)
            #data[name]["energy"] = [entry["energy"] for entry in thermo]
            #data[name]["heat_capacity"] = [entry["heat_capacity"] for entry in thermo]
            #data[name]["mu"] = mu
        with open( OUTFILE, 'w' ) as outfile:
            json.dump( data, outfile, sort_keys=True, indent=2, separators=(",",":") )
        print ( "Data written to {}".format(OUTFILE) )

    if ( comm.Get_size() == 1 ):
        print (ceBulk.atoms.get_chemical_formula() )
        view( ceBulk.atoms )

    ceBulk.atoms.set_calculator(None)
    if ( rank == 0 ):
        with open( pck_file, 'wb') as outfile:
            pck.dump( (ceBulk,calc.get_cf(),calc.eci), outfile )
            print ( "Bulk crystal object pickled to {}".format(pck_file) )

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
