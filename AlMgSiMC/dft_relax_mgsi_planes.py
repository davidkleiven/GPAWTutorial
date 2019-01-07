import sys
from gpaw import GPAW, PW
from ase.io import read
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.build import bulk
import json

def main(fname):
    atoms = read(fname)
    prefix = fname.split("/")[-1]
    traj = Trajectory(fname+".traj", atoms)
    calc = GPAW(mode=PW(600), xc="PBE", nbands="120%", kpts={"density": 5.4, "even": True})
    atoms.set_calculator(calc)
    relaxer = BFGS(atoms, logfile=fname+".log")
    relaxer.attach(traj)
    relaxer.run(fmax=0.025)

def formation_energies():
    form = {}
    a = 4.05
    atoms = bulk("Al", crystalstructure="fcc", a=a)
    calc = GPAW(mode="fd", h=0.18, xc="PBE", nbands=-10, kpts={"density": 5.4, "even": True})
    atoms.set_calculator(calc)
    form["Al"] = atoms.get_potential_energy()

    atoms = bulk("Mg", crystalstructure="fcc", a=a)
    atoms.set_calculator(calc)
    form["Mg"] = atoms.get_potential_energy()

    atoms = bulk("Si", crystalstructure="fcc", a=a)
    atoms.set_calculator(calc)
    form["Si"] = atoms.get_potential_energy()

    fname = "data/form_energies.json"
    with open(fname, 'w') as out:
        json.dump(form, out)


if __name__ == "__main__":
    #main(sys.argv[1])
    formation_energies()