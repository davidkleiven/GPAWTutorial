import sys
import gpaw as gp
from ase.db import connect
from atomtools.ase import delete_vacancies, SaveRestartFiles
from ase.io.trajectory import Trajectory

db_name = "/home/davidkl/GPAWTutorial/CE/AlMgSiX_FCC/almgsiX_fcc.db"

def main(argv):
    uid = int(argv[0])
    kpts_density = float(argv[1])
    lattice_param = float(argv[2])
    relax_atoms = int(argv[3])
    final_structure = int(argv[4])

    db = connect(db_name)
    atoms = db.get(id=uid).toatoms()
    atoms = delete_vacancies(atoms)
    name = db.get(id=uid).name

    # Scale the volume
    pos_scaling = lattice_param/4.05
    cell = atoms.get_cell()
    cell *= pos_scaling
    atoms.set_cell(cell, scale_atoms=True)

    kpt = {"density": kpts_density, "even": True}
    calc = gp.GPAW(h=0.18, kpts=kpt, xc="PBE", nbands="120%")
    atoms.set_calculator(calc)

    if relax_atoms == 0 and final_structure == 0:
        atoms.get_potential_energy()

        # Store the energy of the atoms object with the correct name
        # and lattice parameter
        db.write(atoms, name=name, lattice_param=lattice_param,
                 run_type="lattice_param_estimation")
    elif relax_atoms == 1:
        restart_saver = SaveRestartFiles(calc, name)
        db.update(uid, restart_file=SaveRestartFiles.restart_name(name))
        trajObj = Trajectory("trajectory{}.traj".format(name), 'w', atoms)
        relaxer = BFGS(atoms, logfile="log_{}.txt".format(name))
        relaxer.attach(restart_saver, interval=1)
        relaxer.attach(trajObj)
        relaxer.run(fmax=0.025)
        db.write(atoms, name=name, lattice_param=lattice_param, run_type="geometry_opt", restart_file=SaveRestartFiles.restart_name(name))

if __name__ == "__main__":
    main(sys.argv[1:])
