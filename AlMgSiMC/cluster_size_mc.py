from cemc.mcmc import Montecarlo
import dataset
from cemc.mcmc import SiteOrderParameter, Snapshot
from free_energy_barrier import init_bc
import numpy as np
from mpi4py import MPI
import sys
from ase.io import write
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

workdir = "data/large_cluster"
mc_db_name = workdir + "/nanoparticle_stability.db"


def get_nanoparticle():
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.geometry import get_layers
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = [22, 25, 22]
    lc = 4.05
    atoms = FaceCenteredCubic('Si', surfaces, layers, latticeconstant=lc)
    tags, array = get_layers(atoms, (1, 0, 0))
    for t, atom in zip(tags, atoms):
        if t % 2 == 0:
            atom.symbol = "Mg"
    print(atoms.get_chemical_formula())
    return atoms


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

def plot_order():
    db = dataset.connect("sqlite:///{}".format(mc_db_name))
    tbl = db["cluster_stability"]
    T = []
    order = []
    for row in tbl.find():
        T.append(row["temperature"])
        order.append(row["order_param_mean"])

    srt_indx = np.argsort(T)
    T = np.array(T)
    order = np.array(order)
    T = T[srt_indx]
    order = order[srt_indx]
    n_solutes = 16483
    order /= n_solutes
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(T, order, marker="o")
    plt.show()


def equil_and_relax():
    bc = init_bc(50)
    T = 500
    mc = Montecarlo(bc.atoms, T)
    nanop = get_nanoparticle()
    symbs = insert_nano_particle(bc.atoms.copy(), nanop)
    mc.set_symbols(symbs)
    nsteps = int(1E7)
    camera = Snapshot(atoms=mc.atoms, trajfile=workdir+"/snapshots_equil.traj")
    mc.attach(camera, interval=nsteps/20)
    mc.runMC(steps=nsteps, equil=False)
    write(workdir+"/equillibriated600K.xyz", mc.atoms)
    T = 293
    mc = Montecarlo(bc.atoms, T)
    nsteps = int(1E7)
    camera = Snapshot(atoms=mc.atoms, trajfile=workdir+"/snapshots.traj")
    mc.attach(camera, interval=nsteps/20)
    mc.runMC(steps=nsteps, equil=False)


def run(N, T):
    bc = init_bc(N)
    mc = Montecarlo(bc.atoms, T)
    nanop = get_nanoparticle()
    symbs = insert_nano_particle(bc.atoms.copy(), nanop)
    mc.set_symbols(symbs)
    order_param = SiteOrderParameter(mc.atoms)
    nsteps = int(1E6)
    equil_params = {"mode": "fixed", "window_length": int(1E5)}
    mc.attach(order_param)
    mc.runMC(steps=nsteps, equil=True, equil_params=equil_params)
    thermo = mc.get_thermodynamic()
    mean, std = order_param.get_average()
    thermo["order_param_mean"] = mean
    thermo["order_param_std"] = std
    thermo.update(equil_params)
    db = dataset.connect("sqlite:///{}".format(mc_db_name))
    tbl = db["cluster_stability"]
    tbl.insert(thermo)
    fname = workdir + "/final_structure{}K.xyz".format(T)
    write(fname, mc.atoms)

if __name__ == "__main__":
    equil_and_relax()
    # plot_order()
    # T = int(sys.argv[1])
    # run(50, T)
