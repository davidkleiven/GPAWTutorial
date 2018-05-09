import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
from cemc.mcmc import Montecarlo
from ase.ce import BulkCrystal
from cemc.wanglandau.ce_calculator import get_ce_calc
from mpi4py import MPI
import numpy as np
from ase.db import connect
import json

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

mc_db_name = "data/almgsi_fixed_composition.db"
def run(T,mg_conc,si_conc,precs):
    conc_args = {
        "conc_ratio_min_1":[[64,0,0]],
        "conc_ratio_max_1":[[24,40,0]],
        "conc_ratio_min_2":[[64,0,0]],
        "conc_ratio_max_2":[[22,21,21]]
    }
    orig_spin_dict = {
        "Mg":1.0,
        "Si":-1.0,
        "Al":0.0
    }

    kwargs = {
        "crystalstructure":"fcc",
        "a":4.05,
        "size":[4,4,4],
        "basis_elements":[["Mg","Si","Al"]],
        "conc_args":conc_args,
        "db_name":"data/almgsi.db",
        "max_cluster_size":4
    }
    ceBulk = BulkCrystal( **kwargs )
    ceBulk.spin_dict = orig_spin_dict
    ceBulk.basis_functions = ceBulk._get_basis_functions()
    ceBulk._get_cluster_information()
    eci_file = "data/almgsi_fcc_eci.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    print (ecis)
    #calc = CE( ceBulk, ecis, size=(3,3,3) )
    calc = get_ce_calc( ceBulk, kwargs, ecis, size=[10,10,10] )
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )

    comp = {
        "Mg":mg_conc,
        "Si":si_conc,
        "Al":1.0-mg_conc-si_conc
    }
    calc.set_composition(comp)
    for temp,prec in zip(T,precs):
        print ("Current temperature {}K".format(temp))
        mc_obj = Montecarlo( ceBulk.atoms, temp, mpicomm=comm )
        mode = "prec"
        mc_obj.runMC( mode=mode, prec=prec )
        thermo = mc_obj.get_thermodynamic()
        thermo["temperature"] = temp
        thermo["prec"] = prec
        thermo["internal_energy"] = thermo.pop("energy")
        thermo["converged"] = True

        if ( rank == 0 ):
            db = connect( mc_db_name )
            db.write( ceBulk.atoms, key_value_pairs=thermo )

if __name__ == "__main__":
    T = np.linspace(100,800,20)[::-1]
    precs = np.array([1E-3 for i in range(len(T))])
    #precs[-3:] = 1E-5
    precs[-1] = 1E-6
    mg_conc = float(sys.argv[1])
    si_conc = float(sys.argv[2])
    if ( mg_conc+si_conc < 1.0 and not np.allclose([mg_conc,si_conc],0.0) ):
        run(T,mg_conc,si_conc,precs)
