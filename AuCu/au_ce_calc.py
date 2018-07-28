from ase.db import connect
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.calculators.singlepoint import SinglePointCalculator
from scipy.optimize import minimize

db_name = "data/au_cu_inc.db"
def calc(id):
    db = connect(db_name)
    atoms_orig = db.get_atoms(id=id)
    kvp = db.get(id=id).key_value_pairs
    atoms = atoms_orig.copy()

    # Do linear interpolation of the cell size
    a_au = 4.0782
    a_cu = 3.6149

    symbs = atoms.get_chemical_symbols()
    count = {"Au": 0, "Cu": 0}
    for s in symbs:
        count[s] += 1
    c_au = float(count["Au"])/len(atoms)
    a = a_au*c_au + a_cu*(1.0 - c_au)
    a_orig = 3.9
    cell = atoms.get_cell()
    atoms.set_cell(cell*a/a_orig, scale_atoms=True)
    calc = EMT()
    atoms.set_calculator(calc)
    relaxer = BFGS(atoms)
    relaxer.run(fmax=0.025)
    res = minimize(relax_cell, a, args=(atoms, cell))
    print(res["x"])
    relaxer = BFGS(atoms)
    relaxer.run(fmax=0.025)
    energy = atoms.get_potential_energy()

    del db[id]
    calc = SinglePointCalculator(atoms_orig, energy=energy)
    atoms_orig.set_calculator(calc)
    kvp["converged"] = True
    db.write(atoms_orig, key_value_pairs=kvp)

def relax_cell(a, atoms, orig_cell):
    a_orig = 3.9
    cell = orig_cell*a/a_orig
    atoms.set_cell(cell, scale_atoms=True)
    return atoms.get_potential_energy()

if __name__ == "__main__":
    import sys
    uid = int(sys.argv[1])
    calc(uid)
