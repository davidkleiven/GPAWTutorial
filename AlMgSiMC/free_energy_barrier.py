from cemc.mcmc import PseudoBinaryReactPath, PseudoBinarySGC
from ase.ce import BulkCrystal
from cemc import get_ce_calc
import json
from cemc.mcmc import Snapshot
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def init_bc(N):
    conc_args = {
        "conc_ratio_min_1": [[64, 0, 0]],
        "conc_ratio_max_1": [[24, 40, 0]],
        "conc_ratio_min_2": [[64, 0, 0]],
        "conc_ratio_max_2": [[22, 21, 21]]
    }

    kwargs = {
        "crystalstructure": "fcc",
        "a": 4.05,
        "size": [4, 4, 4],
        "basis_elements": [["Al", "Mg", "Si"]],
        "conc_args": conc_args,
        "db_name": "data/almgsi.db",
        "max_cluster_dia": 4
    }

    ceBulk = BulkCrystal(**kwargs)
    eci_file = "data/almgsi_fcc_eci_newconfig.json"
    with open(eci_file, 'r') as infile:
        ecis = json.load(infile)
    db_name = "large_cell_db{}x{}x{}.db".format(N, N, N)
    calc = get_ce_calc(ceBulk, kwargs, ecis, size=[N, N, N], db_name=db_name)
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator(calc)
    return ceBulk


def run(chem_pot, T, N):
    bc = init_bc(N)

    symbols = ["Al", "Mg", "Si"]
    groups = [{"Al": 2}, {"Mg": 1, "Si": 1}]
    mc = PseudoBinarySGC(bc.atoms, T, symbols=symbols,
                         groups=groups, chem_pot=chem_pot, mpicomm=comm,
                         insert_prob=0.5)
    print(mc.chemical_potential)
    print(bc.basis_functions)
    # camera = Snapshot(atoms=mc.atoms, trajfile="data/window{}_{}.traj".format(N, rank))
    # mc.attach(camera, interval=50000)
    # print(mc.atoms_indx)
    react_path = PseudoBinaryReactPath(mc, react_crd=[0, 400], n_windows=20,
                                       n_bins=10)
    react_path.run(nsteps=10000)
    react_path.save(fname="data/almgsi_barrier_{}.h5".format(N))


if __name__ == "__main__":
    run(-0.75, 400, 10)
