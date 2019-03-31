from ase.build import bulk, stack
from ase.db import connect
from ase.visualize import view
import os
from gpaw import GPAW, PW
from ase.optimize.precon import PreconFIRE
from ase.io.trajectory import Trajectory

DB_NAME = "data/surface_tension_mgsi.db"


def get_mgsi():
    atoms = bulk("Al", cubic=True)
    atoms[0].symbol = "Mg"
    atoms[3].symbol = "Mg"
    atoms[1].symbol = "Si"
    atoms[2].symbol = "Si"
    return atoms


def get_al():
    return bulk("Al", cubic=True)


def construct_db_mixed():
    db = connect(DB_NAME)
    for rep in range(9, 12):
        mgsi = get_mgsi()*(rep, 1, 1)
        al = get_al()*(rep, 1, 1)
        merged = stack(al, mgsi, axis=0)
        db.write(merged, surface=0)


def construct_db_clean():
    db = connect(DB_NAME)
    for rep in range(1, 12):
        mgsi = get_mgsi()*(1, 1, rep)
        al = get_al()*(1, 1, rep)
        merged = stack(al, mgsi, axis=2)
        db.write(merged, surface=1)


def construct_db_mgsi():
    db = connect(DB_NAME)
    for rep in range(1, 12):
        mgsi = get_mgsi()*(1, 1, rep)
        mgsi2 = get_mgsi()
        mgsi2.rotate(90, "x")
        mgsi2.wrap()
        mgsi2 *= (1, 1, rep)
        merged = stack(mgsi, mgsi2, axis=2)
        db.write(merged, surface=2)


def run_dft(uid, density, relax):
    db_name = "/home/davidkl/GPAWTutorial/AlMgSiMC/surface_tension_mgsi.db"
    db = connect(db_name)

    atoms = db.get(id=uid)

    kpts = {"even": True, "density": density}
    calc = GPAW(PW(600), kpts=kpts, xc="PBE", nbands="200%")

    atoms.set_calculator(calc)
    if relax:
        relaxer = PreconFIRE(atoms, logfile="logfile{}.log".format(uid))
        relaxer.attach(calc.write, 1, "backup{}.gpw".format(backup))
        traj = Trajectory(atoms, "trajectory{}.traj".format(uid))
        relaxer.attach(traj)
        relaxer.run(fmax=0.025, smax=0.003)
        db.write(atoms, init_id=uid, runtype="geometry_opt")
    else:
        energy = atoms.get_potential_energy()
        init_id = db.get(id=uid).get("init_id", -1)
        db.write(atoms, init_id=init_id)

if __name__ == "__main__":
    run_dft(int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]))
    