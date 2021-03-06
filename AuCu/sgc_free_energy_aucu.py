from cemc.mcmc import SGCFreeEnergyBarrier
from cemc import get_ce_calc
from mpi4py import MPI
from ase.ce import BulkCrystal
import json

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

def run(chem_pot, min_c1, max_c1, T):
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
        "db_name": 'temp_sgc_{}.db'.format(rank),
        "max_cluster_size": 3,
        "max_cluster_dist": 1.5*alat
        }
    bc = BulkCrystal(**kwargs)
    with open("data/eci_aucu.json", 'r') as infile:
        eci = json.load(infile)
    calc = get_ce_calc(bc, kwargs, eci=eci, size=[10, 10, 10])
    bc = calc.BC
    bc.atoms.set_calculator(calc)
    if rank == 0:
        print("Number of atoms: {}".format(len(bc.atoms)))

    mc = SGCFreeEnergyBarrier(bc.atoms, T, n_windows=10, n_bins=8,
                              min_singlet=min_c1, max_singlet=max_c1,
                              mpicomm=comm, symbols=["Au", "Cu"],
                              save_last_state=True)
    mu = {"c1_0": chem_pot}
    mc.run(nsteps=100000, chem_pot=mu)
    # mc.save(fname="data/barriers/free_eng_{}_{}_{}_{}.json".format(int(min_c1*100), int(max_c1*100), int(1E6*chem_pot), T))
    mc.save(fname="data/barrier_sgc_waste4.json")

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    T = 300
    min_c1 = -1.0
    max_c1 = -0.5
    chem_pots = [0.241207, 0.243333]
    chem_pots = [0.2]
    temps = [300]
    # # for chem_pot in chem_pots:
    # for T in temps:
    #     run(chem_pots[0], min_c1, max_c1, T)
    SGCFreeEnergyBarrier.plot(fname="data/barrier_sgc_waste4.json")
    plt.show()
