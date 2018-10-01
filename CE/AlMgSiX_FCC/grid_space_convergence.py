import gpaw as gp
from ase.db import connect
from ase.build import bulk
import sys

DB_NAME = "/home/davidkl/GPAWTutorial/CE/AlMgSiX/grid_space_conv.db"

def run(h, symb):
    atoms = bulk("Al")
    atoms[0].symbol = symb

    kpts = {"density": 5.4, "even": True}
    calc = GPAW(h=h, kpts=kpts, nbands=-100)
    atoms.set_calculator()
    atoms.get_potential_energy()

    db = connect(DB_NAME)
    db.write(atoms)

if __name__ == "__main__":
    h = float(sys.argv[1])
    symb = sys.argv[2]
    run(h, symb)
