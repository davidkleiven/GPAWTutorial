from gpaw import GPAW, PW
from ase.db import connect
import sys
from ase.optimize import BFGS

DB_NAME = "/home/davidkl/kanzaki.db"

def main(uid):
    print("Solving ID", uid) 
    db = connect(DB_NAME)
    row = db.get(id=uid)
    atoms = row.toatoms()

    calc = GPAW(mode=PW(600), xc="PBE", nbands=-150, kpts={'density': 5.4, 'even': True})
    atoms.set_calculator(calc)
    relaxer = BFGS(atoms, logfile="bfgs{}.log".format(uid))
    relaxer.run(fmax=0.025)
    db.write(atoms, project=row.project, group=row.group)

main(int(sys.argv[1]))
