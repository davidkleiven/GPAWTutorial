import sys
import os
import gpaw as gp
from ase.db import connect
from atomtools.ase import delete_vacancies, SaveRestartFiles
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.optimize.precon import PreconLBFGS, PreconFIRE
from ase.io import read
from ase.constraints import UnitCellFilter

db_name = "/home/davidkl/GPAWTutorial/CE/AlMgSiX_FCC/almgsiX_fcc.db"

def main(argv):
    uid = int(argv[0])

    atoms_from_file = None
    attempt_restart = 1
    kpts_density = 1.37
    lattice_param = 4.05
    relax_atoms = 0
    final_structure = 0
    for arg in argv:
        if arg.find("--atoms=") != -1:
            fname = arg.split("--atoms=")[1]
            atoms_from_file = read(fname)
        elif arg.find("--restart=") != -1:
            attempt_restart = int(arg.split("--restart")[1])
        elif "--kpt=" in arg:
            kpts_density = float(arg.split("--kpt=")[1])
        elif "--a=" in arg:
            lattice_param = float(arg.split("--a=")[1])
        elif "--relax=" in arg:
            relax_atoms = int(arg.split("--relax=")[1])
        elif "--final=" in arg:
            final_structure = int(arg.split("--final=")[1])

    db = connect(db_name)
    atoms = db.get(id=uid).toatoms()
    atoms = delete_vacancies(atoms)
    name = db.get(id=uid).name

    kpt = {"density": kpts_density, "even": True}
    # calc = gp.GPAW(h=0.32, kpts=kpt, xc="PBE", nbands="120%")
    calc = gp.GPAW(mode=gp.PW(600), kpts=kpt, xc="PBE", nbands="120%")
    atoms.set_calculator(calc)
    restart_file = db.get(id=uid).get("restart_file", "")

    if relax_atoms == 0 and final_structure == 0:
        atoms.get_potential_energy()

        # Store the energy of the atoms object with the correct name
        # and lattice parameter
        db.write(atoms, name=name, lattice_param=lattice_param,
                 run_type="lattice_param_estimation")
    elif relax_atoms == 1:
        if os.path.exists(restart_file) and attempt_restart == 1:
            atoms, calc = gp.restart(restart_file)
        else:
            db.update(uid, restart_file=SaveRestartFiles.restart_name(name))
        restart_saver = SaveRestartFiles(calc, name)
        trajObj = Trajectory("trajectory{}.traj".format(name), 'w', atoms)
        ucf = UnitCellFilter(atoms, hydrostatic_strain=True)
        relaxer = PreconFIRE(ucf, logfile="log_{}.txt".format(name))
        relaxer.attach(trajObj)
        relaxer.attach(restart_saver, interval=1)
        relaxer.run(fmax=0.025, smax=0.003)
        db.write(atoms, name=name, lattice_param=lattice_param, run_type="geometry_opt", restart_file=SaveRestartFiles.restart_name(name))

if __name__ == "__main__":
    main(sys.argv[1:])
