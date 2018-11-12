import sys
import os
import gpaw as gp
from ase.db import connect
from atomtools.ase import delete_vacancies, SaveRestartFiles
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.optimize.precon import PreconLBFGS, PreconFIRE
from ase.optimize.sciopt import SciPyFminCG
from ase.parallel import barrier
from ase.io import read
from ase.constraints import UnitCellFilter

db_name = "/home/davidkl/GPAWTutorial/CE/AlMgSiX_FCC/almgsiX_fcc.db"

def main(argv):
    uid = int(argv[0])

    attempt_restart = 1
    kpts_density = 1.37
    lattice_param = 4.05
    relax_atoms = 0
    final_structure = 0
    optimizer = "lbfgs"
    init_from_traj = 0
    for arg in argv:
        if arg.find("--restart=") != -1:
            attempt_restart = int(arg.split("--restart")[1])
        elif "--kpt=" in arg:
            kpts_density = float(arg.split("--kpt=")[1])
        elif "--a=" in arg:
            lattice_param = float(arg.split("--a=")[1])
        elif "--relax=" in arg:
            relax_atoms = int(arg.split("--relax=")[1])
        elif "--final=" in arg:
            final_structure = int(arg.split("--final=")[1])
        elif "--opt=" in arg:
            optimizer = arg.split("--opt=")[1]
        elif "--traj=" in arg:
            init_from_traj = int(arg.split("--traj=")[1])

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
        elif init_from_traj:
            trajfile = "trajectory{}.traj".format(name)
            traj = Trajectory(trajfile, 'r')
            atoms = traj[-1]
            atoms.set_calculator(calc)
        else:
            db.update(uid, restart_file=SaveRestartFiles.restart_name(name))
        restart_saver = SaveRestartFiles(calc, name)
        trajObj = Trajectory("trajectory{}.traj".format(name), 'a', atoms)
        ucf = UnitCellFilter(atoms, hydrostatic_strain=True)
        logfile = "log_{}.txt".format(name)
        if optimizer == "cg":
            relaxer = SciPyFminCG(ucf, logfile=logfile)
        elif optimizer == "fire":
            relaxer = PreconFIRE(ucf, logfile=logfile)
        else:
            relaxer = PreconLBFGS(ucf, logfile=logfile)

        relaxer.attach(trajObj)
        relaxer.attach(restart_saver, interval=1)
        relaxer.run(fmax=0.025)
        db.write(atoms, name=name, lattice_param=lattice_param, run_type="geometry_opt", restart_file=SaveRestartFiles.restart_name(name))
    elif final_structure:
        atoms.get_potential_energy()
        uid = db.write(atoms, name=name, struct_type="final", kpts_density=kpts_density)
        init_id = db.get(name=name, struct_type='initial').id
        db.update(init_id, final_struct_id=uid, converged=1)


if __name__ == "__main__":
    main(sys.argv[1:])
