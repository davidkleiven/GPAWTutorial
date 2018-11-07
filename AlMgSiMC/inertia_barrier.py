from cemc.mcmc import FixedNucleusMC
from cemc.mcmc import Snapshot
from free_energy_barrier import init_bc
from cemc.mcmc import InertiaCrdInitializer, InertiaRangeConstraint
from cemc.mcmc import InertiaBiasPotential
from cemc.mcmc import ReactionPathSampler, AdaptiveBiasReactionPathSampler
from cemc.mcmc import FixEdgeLayers
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
iteration = 0

T = 293

workdir = "data/adaptive_windows{}K".format(T)


def get_nanoparticle():
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.geometry import get_layers
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = [7, 10, 6]
    lc = 4.05
    atoms = FaceCenteredCubic('Si', surfaces, layers, latticeconstant=lc)
    tags, array = get_layers(atoms, (1, 0, 0))
    for t, atom in zip(tags, atoms):
        if t % 2 == 0:
            atom.symbol = "Mg"
    print(atoms.get_chemical_formula())
    return atoms


def update_bias(iter):
    import h5py as h5
    beta = 1.0/(kB*T)
    with h5.File(h5file, 'r') as hfile:
        betaG = np.array(hfile["free_energy"])
        x = np.array(hfile["x"])
        G = betaG/beta

    bias = InertiaBiasPotential(reac_crd=x, free_eng=-G)
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


def insert_nano_particle(atoms, nanoparticle):
    """Insert the nano particle into center of atoms."""
    from scipy.spatial import cKDTree as KDTree
    np_pos = nanoparticle.get_positions()
    com = np.sum(np_pos, axis=0)/len(np_pos)
    np_pos -= com
    nanoparticle.set_positions(np_pos)

    cell = atoms.get_cell()
    diag = 0.5 * (cell[:, 0] + cell[:, 1] + cell[:, 2])
    at_pos = atoms.get_positions() - diag
    tree = KDTree(at_pos)

    used_indices = []
    for atom in nanoparticle:
        dists, closest_indx = tree.query(atom.position)
        if closest_indx in used_indices:
            raise RuntimeError("Two indices map to the same!")
        atoms[closest_indx].symbol = atom.symbol
        used_indices.append(closest_indx)

    symbols = [atom.symbol for atom in atoms]
    return symbols


def run(N, use_bias):
    #network_name=["c2_4p050_3", "c2_2p864_2"]
    bc = init_bc(N)
    mc = FixedNucleusMC(
        bc.atoms, T, network_name=["c2_2p864_2"],
        network_element=["Mg", "Si"], mpicomm=comm,
        max_constraint_attempts=1E6)
    #mc.max_allowed_constraint_pass_attempts = 1
    nanop = get_nanoparticle()
    symbs = insert_nano_particle(bc.atoms.copy(), nanop)
    mc.set_symbols(symbs)
    mc.init_cluster_info()
 
    formula = "(I1+I2)/(2*I3)" # Needle
    #formula = "2*I1/(I2+I3)" # Plate
    inert_init = InertiaCrdInitializer(
        fixed_nucl_mc=mc, matrix_element="Al", cluster_elements=["Mg", "Si"],
        formula=formula)
 
    inert_rng_constraint = InertiaRangeConstraint(
        fixed_nuc_mc=mc, inertia_init=inert_init, verbose=True)

    if use_bias:
        bias = InertiaBiasPotential.load(fname=inertia_bias_file)
        bias.inertia_range = inert_rng_constraint
        mc.add_bias(bias)

    nsteps = 100000

    if rank == 0:
        snap = Snapshot(atoms=bc.atoms, trajfile="{}/inertia.traj".format(workdir))
        mc.attach(snap, interval=100000)
    # reac_path = ReactionPathSampler(
    #     mc_obj=mc, react_crd=[0.05, 0.9], react_crd_init=inert_init,
    #     react_crd_range_constraint=inert_rng_constraint, n_windows=30,
    #     n_bins=10, data_file=h5file)
    # reac_path.run(nsteps=nsteps)
    # reac_path.save()

    inert_rng_constraint.range = [0.0, 0.9]
    fix_layer = FixEdgeLayers(thickness=5.0, atoms=mc.atoms)
    mc.add_constraint(inert_rng_constraint)
    mc.add_constraint(fix_layer)

    reac_path = AdaptiveBiasReactionPathSampler(
        mc_obj=mc, react_crd=[0.0, 0.9], react_crd_init=inert_init, 
        n_bins=100, data_file="{}/adaptive_bias.h5".format(workdir),
        mod_factor=1E-4, delete_db_if_exists=True, mpicomm=None,
        db_struct="{}/adaptive_bias_struct.db".format(workdir))
    reac_path.run()
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
    fig2.savefig(workdir+"/raw_result{}.svg".format(iteration))
    fig.savefig(workdir+"/free_energy{}.svg".format(iteration))
    plt.show()

def plot_iterations(start, end):
    import h5py as h5
    from matplotlib import pyplot as plt
    beta = 1.0 / (kB * T)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    l2_norms = []
    figl2 = plt.figure()
    axl2 = figl2.add_subplot(1, 1, 1)
    for iteration in range(start, end+1):
        fname = workdir + "/inertia_barrier{}.h5".format(iteration)
        with h5.File(fname, 'r') as hfile:
            G = np.array(hfile["free_energy"]) / beta
            x = np.array(hfile["x"])
            l2_norms.append(np.sqrt(np.sum(G**2)))
            print("Datasets in file: {}".format(list(hfile.keys())))
            bias = np.array(hfile["InertiaBiasPotential"])

        ax.plot(x, G-bias, label=str(iteration))
        ax2.plot(x, G, label=str(iteration))
        ax3.plot(x, bias, label=str(iteration))
    ax.set_ylabel("Free energy (eV)")
    ax.set_xlabel("\$\eta = 1 - \\frac{I_1 + I_2}{2I_3}\$")
    ax.legend(loc="best", frameon=False)
    ax2.legend(loc="best", frameon=False)
    axl2.plot(l2_norms, marker="x")
    plt.show()
    # fig2.savefig(workdir+"/raw_result{}.svg".format(iteration))
    # fig.savefig(workdir+"/free_energy{}.svg".format(iteration))


if __name__ == "__main__":
    option = sys.argv[1]
    iter = int(sys.argv[2])
    iteration = iter
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
    elif option == "plot_iter":
        start = int(sys.argv[2])
        end = int(sys.argv[3])
        plot_iterations(start, end)
