from cemc.mcmc import SGCMonteCarlo
from cemc import CE, get_ce_calc
import dataset
from ase.io import read
from ase.ce import BulkCrystal
import json
from mpi4py import MPI
import numpy as np
from cemc.tools import PhaseBoundaryTracker, save_phase_boundary
import copy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

gs = {
    "data/atoms_Au250Cu750.xyz": 55,
    "data/atoms_Au500Cu500.xyz": 110,
    "data/atoms_Au750Cu250.xyz": 165
}

canonical_db = "data/sa_aucu_only_pairs.db"
sgc_db_name = "data/sa_sgc_aucu_new_eci_100T.db"
folder = "data/Au_Au3Cu/"


def get_pure_cf(eci, bf_value):
    cf = {}
    for name in eci.keys():
        size = int(name[1])
        cf_val = bf_value**size
        cf[name] = cf_val
    return cf


def run_mc(phase1, phase2):
    alat = 3.8
    conc_args = {}
    conc_args['conc_ratio_min_1'] = [[1, 0]]
    conc_args['conc_ratio_max_1'] = [[0, 1]]
    kwargs = {
        "crystalstructure": 'fcc',
        "a": 3.8,
        "size": [10, 10, 10],
        # "size": [2, 2, 2],
        "basis_elements": [['Cu', 'Au']],
        "conc_args": conc_args,
        "db_name": 'temp_sgc.db',
        "max_cluster_size": 2,
        "max_cluster_dist": 1.5*alat
        }
    bc1 = BulkCrystal(**kwargs)
    bc2 = copy.deepcopy(bc1)
    bf = bc1._get_basis_functions()[0]

    with open("data/eci_aucu.json", 'r') as infile:
        eci = json.load(infile)

    db = dataset.connect("sqlite:///{}".format(canonical_db))
    tbl = db["corrfunc"]
    if phase1 == "Au" or phase1 == "Cu":
        atoms1 = bc1.atoms
        for atom in atoms1:
            atom.symbol = phase1
        cf1 = get_pure_cf(eci, bf[phase1])
    else:
        atoms1 = read(phase1)
        row = tbl.find_one(runID=gs[phase1])
        row.pop("id")
        row.pop("runID")
        cf1 = row
        bc1.atoms = atoms1

    atoms2 = read(phase2)
    row = tbl.find_one(runID=gs[phase2])
    row.pop("id")
    row.pop("runID")
    cf2 = row
    bc2.atoms = atoms2

    calc1 = CE(bc1, eci=eci, initial_cf=cf1)
    atoms1.set_calculator(calc1)

    calc2 = CE(bc2, eci=eci, initial_cf=cf2)
    atoms2.set_calculator(calc2)

    gs1 = {
        "bc": bc1,
        "eci": eci,
        "cf": cf1
    }

    gs2 = {
        "bc": bc2,
        "eci": eci,
        "cf": cf2
    }
    both_gs = [gs1, gs2]
    mc_args = {
        "mode": "fixed",
        "steps": 100*len(atoms1),
        "equil_params": {"mode": "fixed", "maxiter": 10*len(atoms1)}
    }
    init_mu = None
    # init_mu = [0.271791267568668]
    Tend = None
    # Tend = 100
    for i in range(5, 100):
        tracker = PhaseBoundaryTracker(
            both_gs, backupfile="{}/backup_{}.h5".format(folder, i))
        res = tracker.separation_line_adaptive_euler(
            init_temp=100, stepsize=50, min_step=5.0, mc_args=mc_args,
            symbols=["Au", "Cu"], init_mu=init_mu, Tend=Tend)
        if rank == 0:
            save_phase_boundary("{}/phase_boundary_almg_{}.h5".format(folder, i), res)


def sa_sgc():
    alat = 3.8
    conc_args = {}
    conc_args['conc_ratio_min_1'] = [[1, 0]]
    conc_args['conc_ratio_max_1'] = [[0, 1]]
    kwargs = {
        "crystalstructure": 'fcc',
        "a": 3.8,
        "size": [10, 10, 10],
        # "size": [2, 2, 2],
        "basis_elements": [['Cu', 'Au']],
        "conc_args": conc_args,
        "db_name": 'temp_sgc.db',
        "max_cluster_size": 2,
        "max_cluster_dist": 1.5*alat
        }
    with open("data/eci_aucu.json", 'r') as infile:
        eci = json.load(infile)
    bc = BulkCrystal(**kwargs)
    with open("data/eci_aucu.json", 'r') as infile:
        eci = json.load(infile)
    calc = get_ce_calc(bc, kwargs, eci=eci, size=[10, 10, 10])
    bc = calc.BC
    bc.atoms.set_calculator(calc)
    atoms = bc.atoms

    chem_pot = (np.linspace(0.19, 0.35, 25)+0.00222222222).tolist()
    chem_pot += (np.linspace(0.19, 0.35, 25)+0.00444444444).tolist()
    T = np.linspace(100, 1000, 100)[::-1]
    equil_param = {"mode": "fixed", "maxiter": 10*len(atoms)}
    nsteps = 100*len(atoms)

    sgc_db = dataset.connect("sqlite:///{}".format(sgc_db_name))
    tbl = sgc_db["results"]
    tbl_cf = sgc_db["corr_func"]
    orig_symbs = [atom.symbol for atom in atoms]
    for mu in chem_pot:
        chemical_potential = {"c1_0": mu}
        calc.set_symbols(orig_symbs)
        for temp in T:
            mc = SGCMonteCarlo(atoms, temp, mpicomm=comm, symbols=["Au", "Cu"])
            init_formula = atoms.get_chemical_formula()
            mc.runMC(steps=nsteps, equil_params=equil_param,
                     chem_potential=chemical_potential)
            thermo = mc.get_thermodynamic(reset_ecis=True)
            thermo["init_formula"] = init_formula
            thermo["final_formula"] = atoms.get_chemical_formula()
            if rank == 0:
                uid = tbl.insert(thermo)
                cf = calc.get_cf()
                cf["runID"] = uid
                tbl_cf.insert(cf)

if __name__ == "__main__":
    # run_mc("Au", "data/atoms_Au750Cu250.xyz")
    sa_sgc()
