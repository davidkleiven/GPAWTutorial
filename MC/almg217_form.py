import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
from cemc.mcmc import Montecarlo
from ase.db import connect
from ase.ce import BulkSpacegroup
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
from cemc.mfa import CanonicalMeanField

mc_db_name = "data/almg217_formation_fixed.db"
mf_outfile = "data/mean_field_217.json"
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
run_mfa =True

def run( mg_conc ):
    bs_kwargs = {
    "conc_args":{
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]]
    },
    "basis_elements":[["Al","Mg"],["Al","Mg"],["Al","Mg"],["Al","Mg"]],
    "cellpar":[10.553, 10.553, 10.553, 90, 90, 90],
    "basis":[(0, 0, 0), (0.324, 0.324, 0.324), (0.3582, 0.3582, 0.0393), (0.0954, 0.0954, 0.2725)],
    "spacegroup":217,
    "max_cluster_size":4,
    "db_name":"trial_217.db",
    "size":[1,1,1],
    "grouped_basis":[[0,1,2,3]]
    }
    bs = BulkSpacegroup(**bs_kwargs)
    eci_file = "data/almg_217_eci.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    calc = get_ce_calc( bs, bs_kwargs, ecis, size=[3,3,3] )
    bs = calc.BC
    bs.atoms.set_calculator( calc )
    comp = {
        "Al":1.0-mg_conc,
        "Mg":mg_conc
    }

    calc.set_composition(comp)
    print ("Number of atoms: {}".format(len(bs.atoms)) )
    high_temps = [10000,9000,8000,7000,6000,5000,4000,3000,2000,1000]
    low_temps = range(200,1000,50)[::-1]
    T = np.array(high_temps+low_temps)
    #T = np.array([10000,9000,8000,7000,6000,5000,4000,3000,2000,1000,800,700,600,500,400,375,350,325,300,275,250,225,200,175,150])
    #T = np.array([1E6,100000])
    precs = np.zeros(len(T))+1E-4
    #precs[T<=500] = 1E-5
    print (bs.atoms.get_chemical_formula())
    mc_obj = Montecarlo( bs.atoms, T[0], mpicomm=comm )
    mc_obj.accept_first_trial_move_after_reset = False
    for prec,temp in zip(precs,T):
        mc_obj.T = temp
        mc_obj.reset()
        #if ( temp==T[0] ):
        #    mc_obj.is_first = False # Do not accept the first move

        print ("Current temperature {}K".format(temp))
        mode = "fixed"
        mc_obj.runMC( mode=mode, prec=prec, steps=1000000 )
        thermo = mc_obj.get_thermodynamic()
        thermo["temperature"] = temp
        thermo["prec"] = prec
        thermo["internal_energy"] = thermo.pop("energy")
        thermo["converged"] = True

        if ( rank == 0 ):
            cf = calc.get_cf()
            db = connect( mc_db_name )
            thermo.update(cf)
            db.write( bs.atoms, key_value_pairs=thermo )

    if ( rank == 0 and run_mfa ):
        # At this point a ground state should have been reached
        mfa = CanonicalMeanField( atoms=bs.atoms, T=T )
        mfa.relax()
        res = mfa.calculate()
        formula = bs.atoms.get_chemical_formula()
        fname = "data/mfa217/{}mfa.json".format(formula)
        with open(fname,'w') as outfile:
            json.dump( res, outfile, indent=2, separators=(",",":") )

if __name__  == "__main__":
    mg_conc = float(sys.argv[1])
    run(mg_conc)
