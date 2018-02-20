import sys
sys.path.append("/home/davidkl/Documents/GPAWTutorial/CE")
sys.path.append("/home/davidkl/GPAWTutorial/CE")
import ase.db
from ase.optimize import BFGS
from ase.io.trajectory import Trajectory
import os
import sqlite3 as sq
from ase.optimize.precon.precon import Exp
from ase.calculators.eam import EAM

def change_cell_composition_AlMg( atoms ):
    # Count Mg atoms
    counter = 0
    for atom in atoms:
        if ( atom.symbol == "Mg" ):
            counter += 1
    mg_conc = float(counter)/len(atoms)
    if ( mg_conc < 0.06 ):
        return atoms
    a = 4.0483 + 0.45006*mg_conc # Experimental data
    a0 = 4.05
    scaling = a/a0
    cell = atoms.get_cell()
    cell *= scaling
    atoms.set_cell(cell)
    return atoms

db_name = "almg_eam.db"
def main( argv ):
    runID = int(argv[0])
    print ("Running job: %d"%(runID))
    #db_name = "/home/ntnu/davidkl/Documents/GPAWTutorials/ceTest.db"
    db = ase.db.connect( db_name )

    con = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "SELECT value FROM text_key_values WHERE id=? AND key='name'", (runID,) )
    name = cur.fetchone()[0]
    con.close()

    new_run = not db.get( id=runID ).key_value_pairs["started"]
    # Update the databse
    db.update( runID, started=True, converged=False )

    atoms = db.get_atoms(id=runID)
    if ( new_run==False ):
        atoms = change_cell_composition_AlMg(atoms)

    calc = EAM(potential="/home/davidkl/Documents/EAM/mg-al-set.eam.alloy")
    atoms.set_calculator(calc)

    logfile = "CE_eam/ceEAM%d.log"%(runID)
    traj = "CE_eam/ceEAM%d.traj"%(runID)
    trajObj = Trajectory(traj, 'w', atoms )

    relaxer = BFGS( atoms, logfile=logfile)
    relaxer.attach( trajObj )
    relaxer.run( fmax=0.025 )
    energy = atoms.get_potential_energy()
    row = db.get( id=runID )
    del db[runID]
    kvp = row.key_value_pairs
    runID = db.write( atoms, key_value_pairs=kvp )
    db.update( runID, converged=True )
    print ("Energy: %.2E eV/atom"%(energy/len(atoms)) )

if __name__ == "__main__":
    main( sys.argv[1:] )
