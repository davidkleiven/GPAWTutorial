import sys
import os
import gpaw as gp
from ase.db import connect
from atomtools.ase import delete_vacancies, SaveRestartFiles
from ase.io.trajectory import Trajectory
from ase.optimize import BFGS
from ase.parallel import barrier
from ase.io import read

db_name = "/home/davidkl/GPAWTutorial/CE/AlMgSiX_FCC/almgsiX_fcc.db"

def main(argv):
    uid = int(argv[0])
    kpts_density = float(argv[1])
    lattice_param = float(argv[2])
    relax_atoms = int(argv[3])
    final_structure = int(argv[4])

    atoms_from_file = None
    attempt_restart = 1
    for arg in argv:
        if arg.find("--atoms=") != -1:
            fname = arg.split("--atoms=")[1]
            atoms_from_file = read(fname)
        elif arg.find("--restart=") != -1:
            attempt_restart = int(arg.split("--restart")[1])

    db = connect(db_name)
    atoms = db.get(id=uid).toatoms()
    atoms = delete_vacancies(atoms)
    name = db.get(id=uid).name

    if atoms_from_file is not None:
        assert atoms.get_chemical_formula() == atoms_from_file.get_chemical_formula()
        atoms = atoms_from_file
    else:
        # Scale the volume
        pos_scaling = lattice_param/4.05
        cell = atoms.get_cell()
        cell *= pos_scaling
        atoms.set_cell(cell, scale_atoms=True)

    kpt = {"density": kpts_density, "even": True}
    calc = gp.GPAW(h=0.32, kpts=kpt, xc="PBE", nbands="120%")
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
        barrier()
        restart_saver = SaveRestartFiles(calc, name)
        trajObj = Trajectory("trajectory{}.traj".format(name), 'w', atoms)
        relaxer = BFGS(atoms, logfile="log_{}.txt".format(name))
        relaxer.attach(trajObj)
        relaxer.attach(restart_saver, interval=1)
        relaxer.run(fmax=0.025)
        db.write(atoms, name=name, lattice_param=lattice_param, run_type="geometry_opt", restart_file=SaveRestartFiles.restart_name(name))

if __name__ == "__main__":
    main(sys.argv[1:])
