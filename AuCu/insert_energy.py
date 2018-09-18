from cemc import get_ce_calc
from cemc.mcmc import ActivitySampler
import dataset
from ase.ce import BulkCrystal
import json
from mpi4py import MPI
from random import shuffle

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

db_name = "data/insertion_energy.db"


def insert_energy(conc, T):
        alat = 3.8
        conc_args = {}
        conc_args['conc_ratio_min_1'] = [[1, 0]]
        conc_args['conc_ratio_max_1'] = [[0, 1]]
        kwargs = {
            "crystalstructure": 'fcc',
            "a": 3.8,
            "size": [4, 4, 2],
            "basis_elements": [['Cu', 'Au']],
            "conc_args": conc_args,
            "db_name": 'temp_sgc{}.db'.format(rank),
            "max_cluster_size": 3,
            "max_cluster_dist": 1.5*alat
            }
        bc = BulkCrystal(**kwargs)
        bc.reconfigure_settings()

        with open("data/eci_aucu.json", 'r') as infile:
            eci = json.load(infile)

        names = bc.cluster_names
        for key in eci.keys():
            if key not in names:
                raise ValueError("{} is not in cluster_names".format(key))
        calc = get_ce_calc(bc, kwargs, eci=eci, size=[10, 10, 10])
        bc = calc.BC
        atoms = bc.atoms
        atoms.set_calculator(calc)

        comp = {
            "Au": conc,
            "Cu": 1.0 - conc
        }
        calc.set_composition(comp)
        symbs = [atom.symbol for atom in atoms]
        shuffle(symbs)
        calc.set_symbols(symbs)

        sampler = ActivitySampler(atoms, T, moves=[("Au", "Cu")], mpicomm=comm,
                                  prob_insert_move=0.1)
        equil = {"mode": "fixed", "maxiter": 10 * len(atoms)}
        nsteps = 100 * len(atoms)
        sampler.runMC(steps=nsteps, mode="fixed", equil_params=equil)

        thermo = sampler.get_thermodynamic()
        if rank == 0:
            db = dataset.connect("sqlite:///{}".format(db_name))
            tbl = db["results"]
            tbl.insert(thermo)


if __name__ == "__main__":
    conc = [0.05, 0.1, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
    conc += [0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.95, 1.0]
    T = [300, 400, 500, 600]
    for temp in T:
        for c in conc:
            insert_energy(c, temp)
