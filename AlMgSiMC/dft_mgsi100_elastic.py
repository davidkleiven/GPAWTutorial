from gpaw import GPAW
from gpaw import PW
from ase.io import read, write
from ase.optimize.precon import PreconFIRE
from ase.db import connect

db_name = "/home/davidkl/GPAWTutorial/AlMgSiMC/mgsi100_elastic.db"
def relax():
    fname = "/global/work/davidkl/MgSi100/mgsi_100.xyz"
    atoms = read(fname)

    calc = GPAW(mode=PW(800), xc="PBE", nbands=-60, kpts={"density": 5.4, "even": True})
    atoms.set_calculator(calc)
    relaxer = PreconFIRE(atoms, logfile="mgsi100_relax.log")
    relaxer.run(fmax=0.025, smax=0.003)
    write("mgsi100_fully_relaxed.xyz", atoms)

def single_point_energy(uid):
    db = connect(db_name)
    atoms = db.get(id=uid).toatoms()

    calc = GPAW(mode=PW(800), xc="PBE", nbands=-60, kpts={"density": 5.4, "even": True})
    stress = atoms.get_stress()
    db.write(atoms, init_struct=uid)

if __name__ == "__main__":
    relax()
