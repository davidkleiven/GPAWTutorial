import sys
from gpaw import GPAW, PW
from ase.db import connect

DB_NAME = "/home/davidkl/GPAWTutorial/al3mg_elastic.db"

def main(runID):
    db = connect(DB_NAME)
    atoms = db.get(id=runID).toatoms()

    kpt = {"density": 5.37, "even": True}
    calc = GPAW(mode=PW(800), kpts=kpt, xc="PBE", nbands="130%")
    atoms.set_calculator(calc)
    atoms.get_potential_energy()
    db.write(atoms, init_struct=runID)

if __name__ == "__main__":
    main(sys.argv[1])