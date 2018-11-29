from gpaw import GPAW
from gpaw import PW
from ase.io import read
from ase.optimize.precon import PreconFIRE

def relax():
    fname = "/global/work/davidkl/MgSi100/mgsi_100.xyz"
    atoms = read(fname)

    calc = GPAW(mode=PW(800), xc="PBE", nbands=-60, kpts={"density": 5.4, "even": True})
    atoms.set_calculator(calc)
    relaxer = PreconFIRE(atoms, logfile="mgsi100_relax.log")
    relaxer.run(fmax=0.025, smax=0.003)


if __name__ == "__main__":
    relax()
