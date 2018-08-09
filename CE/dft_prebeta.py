import sys
sys.path.append("/home/davidkl/Documents/GPAWTutorial/CE")
sys.path.append("/home/davidkl/GPAWTutorial/CE")
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS, PreconFIRE
from ase.io.trajectory import Trajectory
import os
from atomtools.ase.save_restart import SaveRestartFiles
from atomtools.ase import delete_vacancies
from ase.calculators.singlepoint import SinglePointCalculator


def main(argv):
    runID = int(argv[0])
    kpt_density = float(argv[1])

    single_point = False
    if len(argv) >= 3:
        single_point = int(argv[2]) == 1
    if len(argv) >= 4:
        optimizer = argv[3]

    print("Running job: {}".format(runID))
    db_paths = ["/home/ntnu/davidkl/GPAWTutorial/CE/pre_beta.db",
                "pre_beta.db", "/home/davidkl/GPAWTutorial/CE/pre_beta.db"]
    for path in db_paths:
        if os.path.isfile(path):
            db_name = path
            break

    db = ase.db.connect(db_name)

    name = db.get(id=runID).key_value_pairs["name"]

    new_run = not db.get(id=runID).key_value_pairs["started"]

    # Update the databse
    db.update(runID, started=True, converged=False)
    db.update(runID, kpt_density=kpt_density)

    atoms = db.get_atoms(id=runID)
    atoms = delete_vacancies(atoms)

    if len(atoms) == 1:
        nbands = -10
    else:
        nbands = "120%"

    kpts = {"density": kpt_density, "even": True}
    try:
        fname = SaveRestartFiles.restart_name(name)
        atoms, calc = gp.restart(fname)
        if kpt_density > 1.5:
            calc = gp.GPAW(mode=gp.PW(600), xc="PBE", kpts=kpts, nbands=nbands)
            atoms.set_calculator(calc)
    except Exception:
        calc = gp.GPAW(mode=gp.PW(600), xc="PBE", kpts=kpts, nbands=nbands)
        atoms.set_calculator(calc)

    logfile = "prebeta{}.log".format(runID)
    traj = "prebeta{}.traj".format(runID)
    trajObj = Trajectory(traj, 'w', atoms)

    restart_saver = SaveRestartFiles(calc, name)

    try:
        fmax = 0.025
        smax = 0.003
        if not single_point:
            if optimizer == "fire":
                relaxer = PreconFIRE(atoms, logfile=logfile)
            else:
                relaxer = PreconLBFGS(
                    atoms,
                    logfile=logfile,
                    use_armijo=True,
                    variable_cell=True)
            relaxer.attach(trajObj)
            relaxer.attach(restart_saver, interval=1)
            relaxer.run(fmax=fmax, smax=smax)

        energy = atoms.get_potential_energy()
        print("Energy: {}".format(energy))

        orig_atoms = db.get_atoms(name=name)
        scalc = SinglePointCalculator(orig_atoms, energy=energy)
        orig_atoms.set_calculator(scalc)
        kvp = db.get(name=name).key_value_pairs
        del db[runID]
        newID = db.write(orig_atoms, key_value_pairs=kvp)
        db.update(newID, converged_stress=True, converged_force=True)

        db.update(newID, single_point=single_point)
        db.update(newID, restart_file=SaveRestartFiles.restart_name(name))
        row = db.get(id=newID)
        conv_force = row.get("converged_force", default=0)
        conv_stress = row.get("converged_stress", default=0)
        if ((conv_force == 1) and (conv_stress == 1) and (kpt_density > 1.5)):
            db.update(newID, converged=True)
    except Exception as exc:
        print(exc)


if __name__ == "__main__":
    main(sys.argv[1:])
