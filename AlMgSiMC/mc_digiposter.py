from cemc.mcmc import PseudoBinarySGC
from ase.clease import CEBulk, Concentration
from cemc import get_atoms_with_ce_calc
from cemc.mcmc import Snapshot, Montecarlo
from ase.clease.tools import wrap_and_sort_by_position
import json
import dataset
import numpy as np
import sys
from ase.io import read
from cemc.tools import FreeEnergy

def get_atoms(cubic=False):
    conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])

    kwargs = {
        "crystalstructure": "fcc",
        "a": 4.05,
        "size": [2, 2, 2],
        "concentration": conc,
        "db_name": "data/almgsi_free_eng.db",
        "max_cluster_size": 4,
        "max_cluster_dia": [7.8, 5.0, 5.0],
        "cubic": cubic
    }
    N = 20
    ceBulk = CEBulk(**kwargs)
    print(ceBulk.basis_functions)
    eci_file = "data/almgsi_fcc_eci.json"
    eci_file = "data/eci_almgsi_aicc.json"
    eci_file = "data/eci_bcs.json"
    eci_file = "data/eci_almgsi_loocv.json"
    eci_file = "data/eci_l1.json"
    with open(eci_file, 'r') as infile:
        ecis = json.load(infile)
    db_name = "large_cell_db_digi{}x{}x{}.db".format(N, N, N)
    atoms = get_atoms_with_ce_calc(ceBulk, kwargs, ecis, size=[N, N, N], 
                                  db_name=db_name)
    return atoms

def main():
    atoms = get_atoms(cubic=True)

    T = 1500
    mc = Montecarlo(atoms, T)
    mc.insert_symbol_random_places("Mg", num=400, swap_symbs=["Al"])
    mc.insert_symbol_random_places("Si", num=400, swap_symbs=["Al"])
    snap = Snapshot(trajfile="data/mc_digi{}K.traj", atoms=atoms)
    mc.attach(snap, interval=50000)
    mc.runMC(steps=500000)

    T = 293
    mc = Montecarlo(atoms, T)
    snap = Snapshot(trajfile="data/mc_digi{}K.traj", atoms=atoms)
    mc.attach(snap, interval=50000)
    mc.runMC(steps=500000)

if __name__ == "__main__":
    main()