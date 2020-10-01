from ase.db import connect
import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
from matplotlib import pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from typing import NamedTuple, List
from ase.visualize import view

REF_ENG = {
    "Al": -3.735496408086985, 
    "Mg": -1.608, 
    "Si": -4.800166410712106,
    "Cu": -3.674
}

sa_db = "data/almgsicu_sa.db"

class NegativeHull(NamedTuple):
    simplices: List[int]
    pts: np.ndarray

def count_atoms(atoms):
    count = {'Al': 0, 'Mg': 0, 'Si': 0, 'Cu': 0}
    for atom in atoms:
        count[atom.symbol] += 1
    return count

def convex_hull(dE, concs):
    pts = np.zeros((len(dE)+1, 4))
    for i in range(len(dE)):
        pts[i, 0] = concs[i]['Al']
        pts[i, 1] = concs[i]['Mg']
        pts[i, 2] = concs[i]['Si']
        pts[i, 3] = dE[i]

    # Add pure Si at last
    pts[-1, 2] = 1.0

    hull = ConvexHull(pts)

    # Remove the ones with positive dE
    simplices = [s for s in hull.simplices if all(pts[x, 3] < 1e-16 for x in s)]
    return NegativeHull(simplices=simplices, pts=pts)

def structs_on_hull(neg_hull: NegativeHull):
    idx  = set()
    for s in neg_hull.simplices:
        idx = idx.union(s)
    return idx

def show_stable(ids):
    atoms = []
    db = connect(sa_db)
    for i in ids:
        atoms.append(db.get(id=int(i)).toatoms())

    view(atoms)

def main():
    concs = []
    dE = []
    ids = []
    with connect(sa_db) as db:
        for row in db.select():
            atoms = row.toatoms()
            count = count_atoms(atoms)
            E = row.energy - sum(REF_ENG[s]*v for s, v in count.items())
            E /= len(atoms)
            dE.append(E)
            concs.append({k: v/len(atoms) for k, v in count.items()})
            ids.append(row.id)

    neg_hull = convex_hull(dE, concs)
    stable_structs = structs_on_hull(neg_hull)
    print(max(stable_structs))
    # show_stable([ids[s] for s in stable_structs if s < len(ids)])

    fig = plt.figure(figsize=(4, 3))
    mg_conc = [c['Cu'] for c in concs]
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(mg_conc, dE, 'o', mfc='none')
    ax.set_xlabel('Concentration')
    ax.set_ylabel("Formation energy (eV/atom)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    plt.show()

main()