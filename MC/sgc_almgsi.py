import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
import os
from cemc.mcmc import SGCMonteCarlo
from ase.ce import BulkCrystal
from cemc.wanglandau.ce_calculator import get_ce_calc
from mpi4py import MPI
import numpy as np
#from ase.db import connect
import json
from cemc.tools import ChemicalPotentialROI
import dataset
from ase.io import Trajectory

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

idun_path="/home/davidkl/AlMgSiCooling"
folders = ["/home/davidkl/AlMgSiCooling","data/"]
mc_db_name = "almgsi_sgc_roi2.db"

for folder in folders:
    if os.path.exists(folders+mc_db_name):
        mc_db_name = folder+mc_db_name
        break


def init_BC():
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
    return ceBulk

def chem_pot_roi():
    ceBulk = init_BC()
    roi = ChemicalPotentialROI(ceBulk.atoms, symbols=["Al","Mg","Si"])
    mg3si = {
        "energy":-2565.33441122719/1000.0,
        "singlets":{"c1_1":-0.706549871364011,"c1_0":0.61161922828621}
    }
    chem_pots = roi.chemical_potential_roi(internal_structure=mg3si)
    sampling, names = roi.suggest_mu(mu_roi=chem_pots, N=5, extend_fraction=0.1)
    db = dataset.connect("sqlite:///"+mc_db_name)
    tbl = db["systems"]
    for line in sampling:
        for row in range(line.shape[0]):
            data = {key:line[row,i] for i,key in enumerate(names)}
            tbl.insert(data)

def run(T,sysID):
    ceBulk = init_BC()

    prec = 1E-4
    db = dataset.connect("sqlite:///"+mc_db_name)
    entry = db["systems"].find_one(id=sysID)
    chem_pot = {"c1_0":entry["c1_0"],"c1_1":entry["c1_1"]}
    if entry["status"] == "finished":
        return
    equil_params = {"window_length":30*len(ceBulk.atoms),"mode":"fixed"}
    max_steps = 1000*len(ceBulk.atoms)
    trajfile = "data/almgsi_sgc/traj_{}.traj".format(sysID)
    traj = None
    if rank == 0:
        traj = Trajectory(trajfile, mode='w')
    for temp in T:
        print ("Current temperature {}K".format(temp))
        mc_obj = SGCMonteCarlo( ceBulk.atoms, temp, mpicomm=comm, symbols=["Al","Mg","Si"] )
        mc_obj.runMC( mode="prec", prec=prec, chem_potential=chem_pot, equil_params=equil_params, steps=max_steps )
        thermo = mc_obj.get_thermodynamic()
        thermo["temperature"] = temp
        thermo["prec"] = prec
        thermo["internal_energy"] = thermo.pop("energy")
        thermo["converged"] = True
        thermo["muc1_0"] = chem_pot["c1_0"]
        thermo["muc1_1"] = chem_pot["c1_1"]

        if ( rank == 0 ):
            thermo["sysID"] = sysID
            newID = db["thermodynamic"].insert(thermo)
            cf = ceBulk.atoms._calc.get_cf()
            cf["resultID"] = newID
            db["correlation"].insert(cf)
            atoms_cpy = ceBulk.atoms.copy()
            atoms_cpy.set_calculator(None)
            traj.write(atoms_cpy)

    if ( rank == 0 ):
        db["systems"].update({"status":"finished","id":sysID}, ["id"])

if __name__ == "__main__":
    T = np.arange(100,2000,50)[::-1]
    sysID = int(sys.argv[1])
    run(T,sysID)
    #chem_pot_roi()
