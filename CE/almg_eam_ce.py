import sys
sys.path.append("/home/davidkl/Documents/GPAWTutorial/CE")
sys.path.append("/home/davidkl/GPAWTutorial/CE")
import ase.db
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
import os
import sqlite3 as sq
from ase.build import bulk
from ase.db import connect
from ase.optimize.precon.precon import Exp
from ase.calculators.eam import EAM
from ase.constraints import StrainFilter
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import minimize

db_name_in = "ce_hydrostatic_eam_relax_effect.db"
db_name_out = "ce_hydrostatic_eam_relax_effect_ideal.db"
db_name_atoms = "ce_hydrostatic_eam_relax_effect_atoms.db"

def create_db_of_ideal_structures():
    atoms = bulk( "Al" )
    atoms = atoms*(4,4,4)
    db_in = connect( db_name_in )
    db_out = connect( db_name_out )
    for row in db_in.select():
        atoms_in = db_in.get_atoms( id=row.id )
        kvp = row.key_value_pairs
        for i in range(len(atoms)):
            atoms[i].symbol = atoms_in[i].symbol
        db_out.write( atoms, key_value_pairs=kvp )

def set_cell_parameter(atoms, a):
    new_atoms = bulk("Al","fcc",a=a)
    new_atoms = new_atoms*(4,4,4)
    for i in range(len(new_atoms)):
        new_atoms[i].symbol = atoms[i].symbol
    new_atoms.set_calculator( atoms._calc )
    return new_atoms

def target_function( a, atoms ):
    new_atoms = set_cell_parameter( atoms, a )
    new_atoms.set_calculator( atoms._calc )
    energy = new_atoms.get_potential_energy()
    return energy

def main( argv ):
    relax_atoms = (argv[1] == "atoms")
    runID = int(argv[0])
    print ("Running job: %d"%(runID))
    db_name = db_name_atoms
    #db_name = "/home/ntnu/davidkl/Documents/GPAWTutorials/ceTest.db"
    db = ase.db.connect( db_name )

    new_run = not db.get( id=runID ).key_value_pairs["started"]
    # Update the databse
    db.update( runID, started=True, converged=False )

    atoms = db.get_atoms(id=runID)

    calc = EAM(potential="/home/davidkl/Documents/EAM/mg-al-set.eam.alloy")
    atoms.set_calculator(calc)
    init_energy = atoms.get_potential_energy()

    logfile = "CE_eam/ceEAM%d.log"%(runID)
    traj = "CE_eam/ceEAM%d.traj"%(runID)
    trajObj = Trajectory(traj, 'w', atoms )

    if ( relax_atoms ):
        relaxer = BFGS( atoms, logfile=logfile)
        relaxer.attach( trajObj )
        relaxer.run( fmax=0.025 )
        energy = atoms.get_potential_energy()
    else:
        res = minimize( target_function, x0=4.05, args=(atoms,) )
        a = res["x"]
        atoms = set_cell_parameter(atoms,a)
        energy = atoms.get_potential_energy()
        print ("Final energy: {}, final a_la: {}".format(energy,a))
    row = db.get( id=runID )
    del db[runID]
    kvp = row.key_value_pairs
    kvp["init_energy"] = init_energy
    runID = db.write( atoms, key_value_pairs=kvp )
    db.update( runID, converged=True )
    print ("Energy: %.2E eV/atom"%(energy/len(atoms)) )
    print ("Initial energy: %.2E eV/atom"%(init_energy/len(atoms)))

if __name__ == "__main__":
    #create_db_of_ideal_structures()
    main( sys.argv[1:] )
