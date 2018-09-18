from cemc.mcmc import FixedNucleusMC
from cemc.mcmc import Snapshot
from free_energy_barrier import init_bc
from cemc.mcmc import InertiaCrdInitializer, InertiaRangeConstraint
from cemc.mcmc import InertiaBiasPotential
from cemc.mcmc import ReactionPathSampler
import numpy as np
from mpi4py import MPI
import sys
from ase.units import kB
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

inertia_bias_file = "inertia_bias_potential.pkl"
inertia_bias_file_new = "inertia_bias_potential.pkl"
h5file = "data/inertia_barrier.h5"

T = 400

workdir = "data/inertia_barrier_full"

def update_bias(iter):
    import h5py as h5
    beta = 1.0/(kB*T)
    with h5.File(h5file, 'r') as hfile:
        G = np.array(hfile["free_energy"])
        x = np.array(hfile["x"])
        betaG = G/beta

    bias = InertiaBiasPotential(reac_crd=x, free_eng=-betaG)
    try:
        old_pot = InertiaBiasPotential.load(fname=inertia_bias_file)
        bias += old_pot
    except Exception:
        pass
    bias.fit_smoothed_curve(smooth_length=21, show=False)
    fig = bias.show()
    fig.savefig("{}/bias_potential{}.svg".format(workdir, iter))
    fig.savefig("{}/bias_potential{}.png".format(workdir, iter))
    bias.save(fname=inertia_bias_file_new)


def run(N, use_bias):
    bc = init_bc(N)
    mc = FixedNucleusMC(
        bc.atoms, T, network_name=["c2_4p050_3", "c2_2p864_2"],
        network_element=["Mg", "Si"], mpicomm=comm)
    mc.max_allowed_constraint_pass_attempts = 1
    mc.insert_symbol_random_places("Mg", swap_symbs=["Al"])
    elements = {
        "Mg": 500,
        "Si": 500
    }

    mc.grow_cluster(elements, shape="sphere", radius=17.0)
    inert_init = InertiaCrdInitializer(
        fixed_nucl_mc=mc, matrix_element="Al", cluster_elements=["Mg", "Si"],
        formula="(I1+I2)/(2*I3)")
    inert_rng_constraint = InertiaRangeConstraint(
        fixed_nuc_mc=mc, inertia_init=inert_init, verbose=True)

    if use_bias:
        bias = InertiaBiasPotential.load(fname=inertia_bias_file)
        bias.inertia_range = inert_rng_constraint
        mc.add_bias(bias)

    nsteps = 100000

    if rank == 0:
        snap = Snapshot(atoms=bc.atoms, trajfile="{}/inertia.traj".format(workdir))
        mc.attach(snap, interval=nsteps/5)
    reac_path = ReactionPathSampler(
        mc_obj=mc, react_crd=[0.0, 1.0], react_crd_init=inert_init,
        react_crd_range_constraint=inert_rng_constraint, n_windows=30,
        n_bins=10, data_file=h5file)
    reac_path.run(nsteps=nsteps)
    reac_path.save()


def plot():
    import h5py as h5
    from matplotlib import pyplot as plt
    beta = 1.0 / (kB * T)
    with h5.File(h5file, 'r') as hfile:
        G = np.array(hfile["free_energy"]) / beta
        x = np.array(hfile["x"])
        print("Datasets in file: {}".format(list(hfile.keys())))
        bias = np.array(hfile["InertiaBiasPotential"])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, G-bias)
    ax.set_ylabel("Free energy (eV)")
    ax.set_xlabel("\$\eta = 1 - \\frac{I_1 + I_2}{2I_3}\$")

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(x, G)
    plt.show()


if __name__ == "__main__":
    option = sys.argv[1]
    iter = int(sys.argv[2])
    inertia_bias_file = "{}/inertia_bias_potential{}.pkl".format(workdir, iter)
    inertia_bias_file_new = "{}/inertia_bias_potential{}.pkl".format(workdir, iter+1)
    h5file = "{}/inertia_barrier{}.h5".format(workdir, iter)
    use_bias = iter > 0
    if option == "run":
        run(50, use_bias)
    elif option == "plot":
        plot()
    elif option == "update_bias":
        update_bias(iter)
