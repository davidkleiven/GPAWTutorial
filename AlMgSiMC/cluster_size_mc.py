from cemc.mcmc import Montecarlo
import dataset
from cemc.mcmc import SiteOrderParameter, Snapshot, MCBackup, EnergyEvolution
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
np_layers = [[22, 25, 22], [21, 24, 21], [20, 23, 20], [19, 22, 19], [18, 21, 18]]
cubic_np_layer = [[18, 18, 5]]

def get_nanoparticle(layer=0):
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.geometry import get_layers
    surfaces = [(1, 0, 0), (1, 1, 0), (1, 1, 1)]
    layers = np_layers[layer]
    lc = 4.05
    atoms = FaceCenteredCubic('Si', surfaces, layers, latticeconstant=lc)
    tags, array = get_layers(atoms, (1, 0, 0))
    for t, atom in zip(tags, atoms):
        if t % 2 == 0:
            atom.symbol = "Mg"
    print(atoms.get_chemical_formula())
    return atoms

def get_cubic_nano_particle(layer=0):
    from ase.cluster.cubic import FaceCenteredCubic
    from ase.geometry import get_layers
    surfaces = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    layers = cubic_np_layer[layer]
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


def equil_and_relax(T, fname="", np_layer=0, nptype="spherical"):
    bc = init_bc(50)
    mc = Montecarlo(bc.atoms, T)
    print ("Running at temperature {}K. Initializing from {}".format(T, fname))
    layer_str = "-".join((str(item) for item in np_layers[np_layer]))
    run_identifier = "{}K_layer{}".format(T, layer_str)
    if fname != "":
        # Initialize atoms object from file
        from ase.io import read
        atoms = read(fname)
        symbs = [atom.symbol for atom in atoms]
        mc.set_symbols(symbs)
        run_identifier = "{}K_{}".format(T, atoms.get_chemical_formula())
    else:
        if nptype == "spherical":
            # Initialize by setting a nano particle at the center
            nanop = get_nanoparticle(layer=np_layer)
        elif nptype == "cubic":
            nanop = get_cubic_nano_particle(layer=np_layer)
            run_identifier += "cubic"
        else:
            raise ValueError("Unknown type {}".format(nanop))
        symbs = insert_nano_particle(bc.atoms.copy(), nanop)
        mc.set_symbols(symbs)
    print("Chemical formula: {}".format(mc.atoms.get_chemical_formula()))
    nsteps = int(4E7)
    camera = Snapshot(atoms=mc.atoms, trajfile=workdir+"/snapshots_equil{}.traj".format(run_identifier))
    energy_evol = EnergyEvolution(mc)
    mc_backup = MCBackup(mc, backup_file=workdir+"/mc_backup{}.pkl".format(run_identifier))
    mc.attach(energy_evol, interval=100000)
    mc.attach(camera, interval=nsteps/20)
    mc.attach(mc_backup, interval=500000)
    mc.runMC(steps=nsteps, equil=False)
    write(workdir+"/equillibriated600K.xyz", mc.atoms)


def extract_largest_cluster(fname):
    from cemc.mcmc import NetworkObserver
    from ase.io import read
    from ase.visualize import view
    bc = init_bc(50)
    calc = bc.atoms.get_calculator()
    atoms = read(fname)
    symbs = [atom.symbol for atom in atoms]
    calc.set_symbols(symbs)
    print(calc.atoms.get_chemical_formula())
    network = NetworkObserver(
        calc=calc, cluster_name=["c2_4p050_3", "c2_2p864_2"],
        element=["Mg", "Si"])

    network(None)
    indices = network.get_indices_of_largest_cluster_with_neighbours()
    # indices = network.get_indices_of_largest_cluster()
    cluster = bc.atoms[indices]
    return cluster

def wulff(fname):
    from ase.io import read, write
    from ase.visualize import view
    from cemc.tools import WulffConstruction
    from matplotlib import pyplot as plt
    atoms = read(fname)
    atoms = extract_largest_cluster(fname)
    surface_file = fname.rpartition(".")[0] + "_onlycluster.xyz"
    wulff = WulffConstruction(cluster=atoms, max_dist_in_element=5.0)
    wulff.filter_neighbours(num_neighbours=9, elements=["Mg", "Si"], cutoff=4.5)
    write(surface_file, wulff.cluster)
    print("Cluster written to {}".format(surface_file))
    surface = wulff.surface_atoms
    view(surface)
    mesh_file = fname.rpartition(".")[0]+"_surfmesh.msh"
    # wulff.fit_harmonics(show=True, order=100, penalty=0.1)
    wulff.interface_energy_poly_expansion(order=12, show=True, spg=225, 
                                          average_cutoff=20.0, penalty=0.1)
    wulff.save_surface_mesh(mesh_file)
    ref_dir = [0.57735027, 0.57735027, -0.57735027]
    gamma = 86.37379010832926/2.0
    theta = np.arccos(ref_dir[2])
    phi = np.arctan2(ref_dir[1], ref_dir[0])
    value = wulff.eval(theta, phi)
    wulff.wulff_plot(show=True, n_angles=60)
    wulff.path_plot(path=[90, 45], normalization=gamma/value)
    wulff.wulff_plot_plane(tol=5.0)
    plt.show()

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
    fname = ""
    T = 400
    layer = 0
    option = "run"
    nptype = "spherical"
    for arg in sys.argv:
        if arg.find("--fname=") != -1:
            fname = arg.split("--fname=")[1]
        elif arg.find("--T=") != -1:
            T = int(arg.split("--T=")[1])
        elif arg.find("--layer=") != -1:
            layer = int(arg.split("--layer=")[1])
        elif arg.find("--option=") != -1:
            option = arg.split("--option=")[1]
        elif arg.find("--nptype=") != -1:
            nptype = arg.split("--nptype=")[1]

    if option == "wulff":
        wulff(fname)
    elif option == "run":
        equil_and_relax(T, fname, np_layer=layer, nptype=nptype)
    # extract_largest_cluster()
    # plot_order()
    # T = int(sys.argv[1])
    # run(50, T)
