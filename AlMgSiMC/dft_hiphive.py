from gpaw import GPAW, PW
from ase.db import connect
import sys
from ase.optimize import BFGS

DB_NAME = "/home/davidkl/mgsi_hiphive.db"


def main(uid):
    print("Solving ID", uid)
    db = connect(DB_NAME)
    row = db.get(id=uid)
    group = row.group
    atoms = row.toatoms()

    calc = GPAW(mode=PW(600), xc="PBE", nbands="150%", kpts={'density': 5.4, 'even': True})
    atoms.set_calculator(calc)
    atoms.get_forces()

    db.write(atoms, group=group, struct_type="final")


main(int(sys.argv[1]))
