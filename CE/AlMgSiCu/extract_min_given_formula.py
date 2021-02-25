from ase.db import connect
import sys
from ase.io import write
import numpy as np
from ase.formula import Formula
from ase.build import find_optimal_cell_shape, make_supercell

def main(target):
    db = connect("data/almgsicu_ce.db")
    energies = []
    atoms = []
    for row in db.select([('converged', '=', 1)]):
        formula = Formula(row.formula)
        form_str = formula.reduce()[0].format('reduce')
        if form_str == target:
            energy = db.get(id=row.final_struct_id).energy/row.natoms
            energies.append(energy)
            atoms.append(row.toatoms())

    if energies:
        idx = np.argmin(energies)
        fname = f"data/cnvx_{target}.cif"
        P = find_optimal_cell_shape(atoms[idx].get_cell(), 32, 'sc', verbose=True)
        cubic = make_supercell(atoms[idx], P)
        write(fname, cubic)
        print(f"Structure written to {fname}")
    else:
        print("No structure matching passed formula")
        

main(sys.argv[1])

