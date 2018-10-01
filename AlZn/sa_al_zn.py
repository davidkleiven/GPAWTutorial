import dataset
from cemc.mcmc import Montecarlo
from cemc import get_ce_calc
import sys
import json
from mpi4py import MPI
from ase.ce import BulkCrystal
from ase.io import write

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

if rank == 0:
    print("Using {} processors".format(comm.Get_size()))

def sa(al_comp, kwargs, eci, db_name, size, lattice):
    comp = {
        "Al": al_comp,
        "Zn": 1.0-al_comp
    }

    bc = BulkCrystal(**kwargs)
    calc = get_ce_calc(bc, kwargs, eci, size)
    bc = calc.BC
    bc.atoms.set_calculator(calc)
    temperatures = [800, 700, 600, 500, 400, 300, 200, 100]
    N = len(bc.atoms)

    if rank == 0:
        print("Supercell has {} atoms".format(N))

    # Define parameters for equillibration
    equil_params = {
        "maxiter": 10 * N,
        "mode": "fixed"
    }

    nsteps = 100 * N
    calc.set_composition(comp)
    for T in temperatures:
        mc = Montecarlo(bc.atoms, T, mpicomm=comm)
        mc.runMC(mode="fixed", steps=nsteps, equil_params=equil_params)
        thermo = mc.get_thermodynamic()
        thermo["converged"] = True
        thermo["al_conc"] = al_comp
        thermo["temperature"] = T
        if rank == 0:
            db = dataset.connect("sqlite:///{}".format(db_name))
            tbl = db["results"]
            tbl.insert(thermo)

    if rank == 0:
        fname = "data/atoms_{}_{}.xyz".format(lattice, bc.atoms.get_chemical_formula())
        write(fname, bc.atoms)


def sa_hcp(al_comp):
    alat = 2.627
    clat = 5.20
    kwargs_hcp = {
        "crystalstructure": "hcp",
        "a": alat,
        "c": clat,
        "size": [4, 4, 2],
        "basis_elements": [["Zn", "Al"]],
        "conc_args": {"conc_ratio_min_1": [[1, 0]],
                      "conc_ratio_max_1": [[0, 1]]},
        "max_cluster_size": 3,
        "max_cluster_dist": alat * 1.5,
        "db_name": "data/zn-al_hcp.db"
    }

    with open("data/eci_alzn_hcp.json") as infile:
        eci = json.load(infile)
    db_name = "data/sa_alzn_hcp.db"
    sa(al_comp, kwargs_hcp, eci, db_name, [10, 10, 5], "hcp")


def sa_fcc(al_comp):
    alat = 4.0
    kwargs_fcc = {
        "crystalstructure": "fcc",
        "a": alat,
        "size": [4, 4, 4],
        "basis_elements": [["Zn", "Al"]],
        "conc_args": {"conc_ratio_min_1": [[1, 0]],
                      "conc_ratio_max_1": [[0, 1]]},
        "max_cluster_size": 4,
        "max_cluster_dist": 1.05 * alat,
        "db_name": "data/zn-al_fcc.db"
    }
    with open("data/eci_alzn_fcc.json") as infile:
        eci = json.load(infile)
    db_name = "data/sa_alzn_fcc_run2.db"
    sa(al_comp, kwargs_fcc, eci, db_name, [10, 10, 10], "fcc")


if __name__ == "__main__":
    comp = float(sys.argv[1])
    sa_fcc(comp)
    # sa_hcp(comp)
