import gpaw as gp
from ase.db import connect
from ase.optimize.precon import PreconLBFGS
from ase.io.trajectory import Trajectory
from ase.calculators.singlepoint import SinglePointCalculator


def main():
    runID = 20  # ID in the SQL database
    kpt_density = 5.4

    print("Running job: {}".format(runID))
    db_name = "database_with_structures.db"

    # Create a database connection
    db = connect(db_name)

    # Retrieve the unique name of this particular entry
    name = db.get(id=runID).key_value_pairs["name"]

    # Update the databse
    db.update(runID, started=True, converged=False)
    db.update(runID, kpt_density=kpt_density)

    # Get the atoms object from the database
    atoms = db.get_atoms(id=runID)
    nbands = "120%"  # Number of electronic bands

    kpts = {"density": kpt_density, "even": True}
    calc = gp.GPAW(mode=gp.PW(600), xc="PBE", kpts=kpts, nbands=nbands)
    atoms.set_calculator(calc)

    logfile = "prebeta{}.log".format(runID)
    traj = "prebeta{}.traj".format(runID)

    # Initialize a trajactory file
    trajObj = Trajectory(traj, 'w', atoms)

    # Initialize the relaxer
    relaxer = PreconLBFGS(atoms, logfile=logfile, use_armijo=True,
                          variable_cell=True)

    # Attach the trajectory file to the relaxer
    relaxer.attach(trajObj)

    # Run until force and stress criteria have been met
    fmax = 0.025  # Maximum force in eV/Å
    smax = 0.003  # Maximum stress in eV/Å^3
    relaxer.run(fmax=fmax, smax=smax)

    # Get and print the total energy
    energy = atoms.get_potential_energy()
    print("Energy: {}".format(energy))

    # What follows is very crucial that it is done exactly like this

    # Retrieve the original (unrelaxed object) from the database
    orig_atoms = db.get_atoms(id=runID)

    # Attacha singlet point calculator with the energy of the relaxed structure
    scalc = SinglePointCalculator(orig_atoms, energy=energy)
    orig_atoms.set_calculator(scalc)

    # Get a all the key_value_pairs
    kvp = db.get(id=runID).key_value_pairs

    # Delete the original entry
    del db[runID]

    # Write the new objet to the database
    # Unrelaxed system, with the energy of the relaxed one
    newID = db.write(orig_atoms, key_value_pairs=kvp)

    # Update the converged flag
    db.update(newID, converged=True)

    # Store also the relaxed object (NOTE: they are linked via their name)
    db.write(atoms, name=name, state="relaxed")


if __name__ == "__main__":
    main()
