import sys
from gpaw import GPAW, PW
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS

def main(fname):
    atoms = read(fname)
    traj = Trajectory(fname+".traj", atoms)
    calc = GPAW(mode=PW(600), xc="PBE", nbands="120%", kpts={"density": 5.4, "even": True})
    atoms.set_calculator(calc)
    relaxer = BFGS(atoms, logfile=fname+".log")
    relaxer.attach(traj)
    relaxer.run(fmax=0.025)

if __name__ == "__main__":
    main(sys.argv[1])