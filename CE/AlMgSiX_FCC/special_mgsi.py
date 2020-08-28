"""
Creates some special/interesting L10 phases to use together with singleVacancy.py
"""
from ase.build import bulk
from ase.geometry import get_layers
import numpy as np
from ase.visualize import view
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
from itertools import product
from ase.io.trajectory import TrajectoryWriter

N = 12

def removeAl(atoms):
    for i in range(len(atoms), 0, -1):
        if atoms[i-1].symbol == 'Al':
            del atoms[i-1]
    return atoms

def main():
    a = 4.05
    atoms = bulk('Al', cubic=True, a=a)*(N, N, N)
    tags, _ = get_layers(atoms, (0, 0, 1))
    pos = atoms.get_positions().copy()
    center = np.mean(pos, axis=0) + np.array([a/4, a/4, 0.0])
    pos -= center
    radii = np.sqrt(np.sum(pos[:, :2]**2, axis=1))
    indices = np.argsort(radii)

    num_per_layer = 5
    all_indices = []
    for layer in range(2, 12):
        symbol = 'Mg' if layer%2 == 0 else 'Si'
        num_inserted = 0
        for i in indices:
            if tags[i] == layer and num_inserted < num_per_layer:
                atoms[i].symbol = symbol
                num_inserted += 1
                all_indices.append(i)

    # Extract all items in the 2 layers above
    indices_si = []
    indices_mg = []
    num_si = 4
    num_mg = 5
    for layer in range(12, 14):
        num_extracted = 0
        num_to_extract = num_mg if layer%2 == 0 else num_si
        for i in indices:
            if tags[i] == layer and num_extracted < num_to_extract:
                if layer % 2 == 0:
                    indices_mg.append(i)
                else:
                    indices_si.append(i)
                num_extracted += 1
    
    print(len(indices_mg), len(indices_si))
    traj = TrajectoryWriter("data/specialMgSiClusters.traj")
    for symbMg in product(['Al', 'Mg'], repeat=len(indices_mg)):
        for symbSi in product(['Al', 'Si'], repeat=len(indices_si)):
            atoms_cpy = atoms.copy()
            for i in range(len(indices_mg)):
                atoms_cpy[indices_mg[i]].symbol = symbMg[i]
                for j in range(len(indices_si)): 
                    atoms_cpy[indices_si[j]].symbol = symbSi[j]
            traj.write(atoms_cpy)
        
   


main()