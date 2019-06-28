from ase.build import bulk
from ase.visualize import view
from ase.db import connect
from ase.io import read

DB_NAME = "prebeta_vibrations.db"
def beta_double_prime(shift):
    atoms = bulk("Al", crystalstructure='fcc', a=4.05, cubic=True)*(4, 4, 1)

    if shift:
        atoms[23].position[2] = 2.025
    view(atoms)


def prepare_db():
    db = connect(DB_NAME)

    atoms = read("data/beta_fcc.xyz")
    db.write(atoms, tag='prebeta_fcc_ideal')
    atoms = read("data/prebeta_shift.xyz")
    db.write(atoms, tag='prebeta_shift_ideal')

prepare_db()
#beta_double_prime(True)