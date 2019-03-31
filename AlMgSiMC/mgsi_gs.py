from cemc.tools import GSFinder
from ase.clease import CEBulk, Concentration
from cemc import get_atoms_with_ce_calc
from ase.clease.tools import wrap_and_sort_by_position
import json
import dataset
import numpy as np
import sys
from ase.io import read, write

def get_atoms():
    conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])

    kwargs = {
        "crystalstructure": "fcc",
        "a": 4.05,
        "size": [2, 2, 2],
        "concentration": conc,
        "db_name": "data/almgsi_free_eng.db",
        "max_cluster_size": 4,
        "cubic": True
    }
    N = 8
    ceBulk = CEBulk(**kwargs)
    print(ceBulk.basis_functions)
    eci_file = "data/almgsi_fcc_eci.json"
    eci_file = "data/eci_bcs.json"
    #eci_file = "data/eci_almgsi_l2.json"
    with open(eci_file, 'r') as infile:
        ecis = json.load(infile)
    db_name = "large_cell_db{}x{}x{}_cubic.db".format(N, N, N)
    atoms = get_atoms_with_ce_calc(ceBulk, kwargs, ecis, size=[N, N, N], 
                                   db_name=db_name)
    return atoms

def main():
    folder = "/home/gudrun/davidkl/Documents/GPAWTutorial/AlMgSiMC/data"
    fname = folder + "/MgSi_mgsi_free_eng.xyz"
    atoms_read = wrap_and_sort_by_position(read(fname))
    db_energy = -218.272/64

    symbols = [atom.symbol for atom in atoms_read]


    T = np.linspace(100.0, 1300.0, 20)[::-1].tolist()
    #T = [300]
    atoms = get_atoms()
    symbols = ["Mg"]*1024 + ["Si"]*1024
    atoms.get_calculator().set_symbols(symbols)
    print("Init energy: {}".format(atoms.get_calculator().get_energy()))
    print(db_energy, atoms.get_calculator().get_energy()/len(atoms))
    #assert abs(db_energy - atoms.get_calculator().get_energy()/len(atoms)) < 1E-3

    gs = GSFinder()
    bc = atoms.get_calculator().BC
    res = gs.get_gs(bc, temps=T, n_steps_per_temp=1000000, atoms=atoms)
    print(res["energy"])
    write("data/mgsi_gs_bcs.xyz", res["atoms"])

if __name__ == "__main__":
    main()
