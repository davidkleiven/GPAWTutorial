from ase.db import connect
from ase.build import bulk
from itertools import product
from random import randint
from clease.tools import all_integer_transform_matrices
from ase.build import make_supercell
from random import choice
from pyxtalcomp import XtalCompASE

db_name = "data/cupd.db"

symbols = ['Cu', 'Pd']

def valid(atoms):
    count = {}
    for atom in atoms:
        count[atom.symbol] = count.get(atom.symbol, 0) + 1
    return count.get('Si', 0) < 0.55*len(atoms)


def create_data_all_comb(atoms):
    all_atoms = []
    for syms in product(symbols, repeat=len(atoms)):
        new_at = atoms.copy()
        new_at.symbols = syms
        if valid(new_at):
            all_atoms.append(new_at)
    return all_atoms

def create_data_random_comb(atoms, num=40):
    all_atoms = []
    for _ in range(num):
        new_at = atoms.copy()
        for atom in new_at:
            atom.symbol = choice(symbols)
        if valid(new_at):
            all_atoms.append(new_at)
    return all_atoms

def all_atoms():
    all_atoms = []
    db = connect(db_name)
    for row in db.select([('struct_type', '=', 'initial')]):
        all_atoms.append(row.toatoms())
    return all_atoms

def exists(a, other):
    comp = XtalCompASE()
    for b in other:
        if comp(a, b):
            return True
    return False

def add_structures(template):
    #new_atoms = create_data_all_comb(template)
    new_atoms = create_data_random_comb(template, num=500)
    in_db = all_atoms()
    #checker = SymmetryEquivalenceCheck(to_primitive=True)

    # Filter only new
    filtered_atoms = []
    for i, atoms in enumerate(new_atoms):
        if not exists(atoms, new_atoms[i+1:]):
            filtered_atoms.append(atoms)

    # Filter based on DB
    filtered_atoms = [a for a in filtered_atoms if not exists(a, in_db)]
    
    with connect(db_name) as db:
        for a in filtered_atoms:
            db.write(a, struct_type='initial', group=randint(0, 2**32-1))
    print(f"Inserted {len(filtered_atoms)} structures")

# for P in all_integer_transform_matrices(5):
#     print(P)
#     prim = bulk('Al', a=4.05, crystalstructure='fcc')
#     template = make_supercell(prim, P)
#     add_structures(template)

prim = bulk('Al', a=4.05, crystalstructure='fcc', cubic=True)*(2, 2, 2)
add_structures(prim)
