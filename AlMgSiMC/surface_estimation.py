import sys
from cemc.mcmc import Montecarlo
from ase.build import cut, bulk
from ase.geometry import get_layers
from ase.ce.tools import wrap_and_sort_by_position
from ase.ce import BulkCrystal
import json
from cemc import get_ce_calc
from cemc.mcmc import MCBackup, Snapshot, EnergyEvolution

WORKDIR = "data/large_cluster/"
def create_surface(sc_shape):
    from scipy.spatial import cKDTree as KDTree
    atoms = bulk("Al")*sc_shape
    if sc_shape[2]%2 != 0:
        raise ValueError("The third direction has to be divisible by 2!")
    num_layers = int(sc_shape[2]/2)

    # Create a cut such that a3 vector is the normal vector
    slab = cut(atoms, a=(1, 0, 0), b=(0, 1, 0), c=(0, 0, 1), nlayers=num_layers)
    tags, array = get_layers(slab, (1, -1, 0)) 
    for tag, atom in zip(tags, slab):
        if tag%2 == 0:
            atom.symbol = "Mg"
        else:
            atom.symbol = "Si"
    tree = KDTree(atoms.get_positions())
    used_indices = []
    for atom in slab:
        dists, closest_indx = tree.query(atom.position)
        if closest_indx in used_indices:
            raise RuntimeError("Two atoms are mapped onto the same index!")
        atoms[closest_indx].symbol = atom.symbol
        used_indices.append(closest_indx)
    return atoms

def main(size, T):
    atoms = create_surface(size)
    atoms = wrap_and_sort_by_position(atoms)
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
    db_name = "large_cell_db{}x{}x{}.db".format(size[0], size[1], size[2])
    calc = get_ce_calc(ceBulk, kwargs, ecis, size=size, db_name=db_name)
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator(calc)
    
    mc = Montecarlo(ceBulk.atoms, T)
    symbs = [atom.symbol for atom in atoms]
    mc.set_symbols(symbs)

    # Write a copy of the current atoms object
    from ase.io import write
    shape_str = "-".join((str(item) for item in size))
    uid_str = "{}K_{}".format(T, shape_str)
    write(WORKDIR+"initial_config{}.xyz".format(uid_str), mc.atoms)
    
    backup = MCBackup(mc, backup_file=WORKDIR+"backup{}.xyz".format(uid_str), 
                      db_name=WORKDIR+"mc_surface.db")
    camera = Snapshot(atoms=mc.atoms, trajfile=WORKDIR+"surface{}.traj".format(uid_str))
    evol = EnergyEvolution(mc)
    nsteps = int(10E6)
    mc.attach(backup, interval=100000)
    mc.attach(camera, interval=int(nsteps/20))
    mc.attach(evol, interval=10*len(mc.atoms))
    mc.runMC(mode="fixed", steps=nsteps, equil=False)
    write(WORKDIR+"final_config{}.xyz".format(uid_str), mc.atoms)

if __name__ == "__main__":
    shape = (10, 10, 10)
    T = 400
    for arg in sys.argv:
        if arg.find("--size=") != -1:
            split = arg.split("--size=")[1].split(",")
            shape = tuple((int(item) for item in split))
        elif arg.find("--T=") != -1:
            T = int(arg.split("--T=")[1])
    print(T)
    print(shape)
    main(shape, T)
