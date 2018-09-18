from cemc.mcmc import PseudoBinaryReactPath, PseudoBinarySGC
from cemc.mcmc import PseudoBinaryFreeEnergyBias
from ase.ce import BulkCrystal
from cemc import get_ce_calc
import json
from cemc.mcmc import Snapshot
from ase.units import kB
import numpy as np
from mpi4py import MPI
import matplotlib as mpl
from matplotlib import pyplot as plt
import h5py as h5

rcParams = {"axes.unicode_minus": False, "svg.fonttype": "none",
            "font.size": 18}
mpl.rcParams.update(rcParams)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def create_bias():
    import h5py as h5
    T = 400
    beta = 1.0/(kB*T)
    data_fname = "data/almgsi_barrier_bias_10.h5"
    with h5.File(data_fname, 'r') as infile:
        reac_crd = np.array(infile["x"])
        beta_G = np.array(infile["free_energy"])/beta

    potential = PseudoBinaryFreeEnergyBias(reac_crd=reac_crd, free_eng=-beta_G)
    potential.fit_smoothed_curve(smooth_length=41, show=True)
    try:
        old_pot = PseudoBinaryFreeEnergyBias.load("data/free_energy_bias.pkl")
        potential += old_pot
        potential.show()
    except Exception as exc:
        print(str(exc))
    potential.fit_smoothed_curve(smooth_length=41, show=True)
    # potential.save(fname="data/free_energy_bias.pkl")


def show_barrier(fname):
    with h5.File(fname, 'r') as hfile:
        print(list(hfile.keys()))
        x = np.array(hfile["x"])
        betaG = np.array(hfile["free_energy"])
        beta = 1.0/(kB * 400)
        try:
            bias_pot = np.array(hfile["PseudoBinaryFreeEnergyBias"])
        except Exception:
            bias_pot = np.zeros_like(betaG)
        print(beta * bias_pot)
        betaG -= beta * bias_pot
        # bias = np.array(hfile["bias_"])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, betaG)
    ax.set_xlabel("Number of MgSi units")
    ax.set_ylabel("\$\Delta G/k_B T\$")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

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
        "max_cluster_size": 4
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
    bias = PseudoBinaryFreeEnergyBias.load(fname="data/free_energy_bias.pkl")
    mc = PseudoBinarySGC(bc.atoms, T, symbols=symbols,
                         groups=groups, chem_pot=chem_pot, mpicomm=comm,
                         insert_prob=0.5)
    mc.add_bias(bias)
    print(mc.chemical_potential)
    print(bc.basis_functions)
    # camera = Snapshot(atoms=mc.atoms, trajfile="data/window{}_{}.traj".format(N, rank))
    # mc.attach(camera, interval=50000)
    # print(mc.atoms_indx)
    react_path = PseudoBinaryReactPath(
        mc, react_crd=[0, 400], n_windows=20, n_bins=10,
        data_file="data/almgsi_barrier_bias_{}.h5".format(N))
    react_path.run(nsteps=500000)


if __name__ == "__main__":
    # run(-0.75, 400, 10)
    # create_bias()
    show_barrier("data/almgsi_barrier_bias_10.h5")
