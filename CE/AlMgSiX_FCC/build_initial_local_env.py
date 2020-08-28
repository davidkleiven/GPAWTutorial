from ase.build import bulk
from ase.geometry import get_layers
from ase.build import stack
from clease.tools import wrap_and_sort_by_position
from ase.io import write

ALTERNATING = "alternating"
SILICON = "silicon"
MAGNESIUM = "magnesium"

def mgsi(interface):
    atoms = bulk('Al', cubic=True)
    mgsi = atoms.copy()

    miller = (0, 0, 1)
    if interface == SILICON or interface == MAGNESIUM:
        miller = (1, 0, 0)

    tags, levels = get_layers(mgsi, miller)
    for i, atom in enumerate(mgsi):
        if interface == SILICON or interface == ALTERNATING:
            if tags[i] % 2 == 0:
                atom.symbol = 'Mg'
            else:
                atom.symbol = 'Si'
        elif interface == MAGNESIUM:
            if tags[i] % 2 == 0:
                atom.symbol = 'Si'
            else:
                atom.symbol = 'Mg'

    # Tag all atoms to -1
    for atom in mgsi:
        atom.tag = -1
    
    mgsi = mgsi*(4, 4, 4)
    matrix = atoms.copy()
    for atom in matrix:
        atom.tag = -1
    matrix = matrix*(4, 4, 4)
    active_area = atoms.copy()
    active_area = active_area*(4, 4, 4)
    tags, layers = get_layers(active_area, (1, 0, 0))
    for t, atom in zip(tags, active_area):
        atom.tag = t

    # Stack togeter
    atoms = stack(mgsi, active_area, axis=0)
    atoms = stack(atoms, matrix, axis=0)
    atoms = wrap_and_sort_by_position(atoms)
    write(f"data/mgsi_active_matrix_{interface}_interface.xyz", atoms)

mgsi(SILICON)

