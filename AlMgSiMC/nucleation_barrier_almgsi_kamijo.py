from ase.clease import CEBulk, Concentration
from cemc import get_atoms_with_ce_calc
import json
import dataset
import numpy as np
from scipy.spatial import cKDTree as KDTree
from ase.build import bulk
from ase.geometry import get_layers
import dataset
from cemc.mcmc import Montecarlo, Snapshot

def get_atoms(cubic=True):
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
    N = 10
    ceBulk = CEBulk(**kwargs)
    print(ceBulk.basis_functions)
    eci_file = "data/almgsi_fcc_eci.json"
    eci_file = "data/eci_almgsi_aicc.json"
    eci_file = "data/eci_bcs.json"
    eci_file = "data/eci_almgsi_loocv.json"
    eci_file = "data/eci_l1.json"
    with open(eci_file, 'r') as infile:
        ecis = json.load(infile)
    db_name = "large_cell_db{}x{}x{}.db".format(N, N, N)
    atoms = get_atoms_with_ce_calc(ceBulk, kwargs, ecis, size=[N, N, N], 
                                   db_name=db_name)
    return atoms

def insert_np(size, atoms):
    cluster = bulk("Al", cubic=True)*(size, size, size)
    tags, array = get_layers(cluster, (1, 0, 0))

    for atom in cluster:
        if atom.index % 2 == 0:
            atom.symbol = "Si"
        else:
            atom.symbol = "Mg"

    tree = KDTree(atoms.get_positions())
    symbols = [a.symbol for a in atoms]
    for atom in cluster:
        closest = tree.query(atom.position)[1]
        symbols[closest] = atom.symbol
    atoms.get_calculator().set_symbols(symbols)
    return atoms

def main():
    energies = []
    sizes = list(range(1, 6))
    atoms = get_atoms()
    insert_np(6, atoms)

    mc = Montecarlo(atoms, 0.1)
    camera = Snapshot(atoms=mc.atoms, trajfile="/work/sophus/nuc_cluster.traj")
    db = dataset.connect("sqlite:////work/sophus/mgsi_nuc_barrier_kamijo.db")
    tbl = db["systems"]
    mc.attach(camera, interval=100*len(atoms))
    num_mg = sum(1 for atom in atoms if atom.symbol == "Mg")
    while num_mg > 2:

        print(atoms.get_chemical_formula())
        mc.runMC(mode="fixed", equil=False, steps=100*len(atoms))
        thermo = mc.get_thermodynamic()
        tbl.insert(thermo)

        # Remove one Mg atom and one Si atom
        symbols = [a.symbol for a in atoms]
        for i in range(20):
            i = symbols.index("Si")
            symbols[i] = "Al"
            i = symbols.index("Mg")
            symbols[i] = "Al"
        mc.set_symbols(symbols)
        num_mg = sum(1 for atom in atoms if atom.symbol == "Mg")

if __name__ == "__main__":
    main()




