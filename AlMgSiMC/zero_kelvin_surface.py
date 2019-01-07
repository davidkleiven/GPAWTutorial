from ase.build import bulk
from ase.clease.tools import wrap_and_sort_by_position
from ase.clease import CEBulk
from cemc import get_ce_calc
from ase.geometry import wrap_positions
from ase.visualize import view
import json
import numpy as np
from ase.units import J, m

def get_mgsi():
    atoms = bulk("Al")
    atoms = atoms*(2, 2, 2)
    si_indx = [1, 3, 4, 6]
    for atom in atoms:
        if atom.index in si_indx:
            atom.symbol = "Si"
        else:
            atom.symbol = "Mg"
    return atoms

def get_mgsi_surface100_si_si():
    from ase.build import fcc100
    atoms = fcc100("Al", size=(10, 10, 5))
    for atom in atoms:
        if atom.tag%2 == 0:
            atom.symbol = "Mg"
        else:
            atom.symbol = "Si"
    return atoms



def surface_energy():
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

    ceBulk = CEBulk(**kwargs)
    eci_file = "data/almgsi_fcc_eci_newconfig.json"
    with open(eci_file, 'r') as infile:
        ecis = json.load(infile)
    size = (8, 8, 8)
    db_name = "large_cell_db{}x{}x{}.db".format(size[0], size[1], size[2])
    calc = get_ce_calc(ceBulk, kwargs, ecis, size=size, db_name=db_name)
    print(calc.get_energy()/len(calc.atoms))
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator(calc)

    # Get the energy of MgSi
    mgsi = wrap_and_sort_by_position(get_mgsi()*(4, 4, 4))
    symbs = [atom.symbol for atom in mgsi]
    calc.set_symbols(symbs)
    view(calc.atoms)
    mgsi_energy = calc.get_energy()/len(calc.atoms)
    print("Energy MgSi: {} eV/atom".format(mgsi_energy))

    # Get the energy of pure al
    symbs = ["Al" for _ in range(len(mgsi))]
    calc.set_symbols(symbs)
    view(calc.atoms)
    al_energy = calc.get_energy()/len(calc.atoms)
    print("Energy Al: {} eV/atom".format(al_energy))

    # Get the energy of a 50% mixture
    mgsi = wrap_and_sort_by_position(get_mgsi()*(4, 4, 2))
    # mgsi = get_mgsi_surface100_si_si()
    from scipy.spatial import cKDTree as KDTree
    cell = calc.atoms.get_cell() 
    tree = KDTree(calc.atoms.get_positions())
    symbs = [atom.symbol for atom in calc.atoms]
    for atom in mgsi:
        pos = np.zeros((1, 3))
        pos[0, :] = atom.position
        wrapped = wrap_positions(pos, cell)
        if not np.allclose(wrapped, atom.position):
            continue
        _, indx = tree.query(atom.position)
        symbs[indx] = atom.symbol
    calc.set_symbols(symbs)
    view(calc.atoms)

    # Surface energy
    mix_energy = calc.get_energy()
    print("Mix energy: {} eV ({} eV/atom)".format(mix_energy, mix_energy/len(calc.atoms)))
    num_al = 0
    num_mgsi = 0
    for atom in calc.atoms:
        if atom.symbol == "Al":
            num_al += 1
        elif atom.symbol == "Mg" or atom.symbol == "Si":
            num_mgsi += 1
    assert num_al + num_mgsi == len(calc.atoms)
    dE = mix_energy - num_al*al_energy - num_mgsi*mgsi_energy
    cell = calc.atoms.get_cell()
    a1 = cell[0, :]
    a2 = cell[1, :]
    normal = np.cross(a1, a2)
    area = np.sqrt(normal.dot(normal))
    
    # Convert units
    surface_tension = dE/area
    mJ = J/1000.0 # milli joules
    surface_tension *= (m*m/mJ)
    unit_normal = normal/area
    print("Surface tension: {} mJ/m^2".format(surface_tension))
    print("Direction: {}".format(unit_normal))

if __name__ == "__main__":
    surface_energy()
    # get_mgsi_surface100()


