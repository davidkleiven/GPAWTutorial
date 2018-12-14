from gpaw import GPAW
from ase.build import bulk
from ase.geometry import get_layers
from ase.io import write

def create_matsudada():
    atoms = bulk("Al", cubic=True, a=4.05)
    atoms = atoms*(5, 5, 5)

    # Construct Matsudada structure
    layer_100, _ = get_layers(atoms, (1, 0, 0))
    indices = [atom.index for atom in atoms if layer_100[atom.index] == 0]
    layer_010, _ = get_layers(atoms, (0, 1, 0))
    for indx in indices:
        if layer_010[indx]%2 == 0:
            atoms[indx].symbol = "Mg"
        else:
            atoms[indx].symbol = "Si"
    
    fname = "data/mgsi_matsudada.xyz"
    write(fname, atoms)
    print("Structure written to {}".format(fname))

def create_linear_elastic():
    atoms = bulk("Al", cubic=True, a=4.05)
    atoms = atoms*(5, 5, 5)

    # Construct structure predicted by linear elasticity
    layer_101, _ = get_layers(atoms, (1, 0, 1))
    mg_indices = [atom.index for atom in atoms if layer_101[atom.index] == 8]
    si_indices = [atom.index for atom in atoms if layer_101[atom.index] == 9]
    for indx in mg_indices:
        atoms[indx].symbol = "Mg"

    fname = "data/mgsi_linear_elastic.xyz"
    write(fname, atoms)
    print("Structure written to {}".format(fname))


create_linear_elastic()
create_matsudada()