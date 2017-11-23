import sys
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from ase.io.trajectory import Trajectory
import os
import sqlite3 as sq
from ase.optimize.precon.precon import Exp

class SaveToDB(object):
    def __init__(self, db_name, runID, name):
        self.db = ase.db.connect( db_name )
        self.runID = runID
        self.name = name
        self.smallestEnergy = 1000.0
        self._is_first = True

    def __call__(self, atoms=None):
        """
        Saves the current run to db if the energy is lower
        """
        if ( atoms is None ):
            return

        if ( self._is_first ):
            self.db.update( self.runID, collapsed=False )
            self._is_first = False

        if ( atoms.get_potential_energy() < self.smallestEnergy ):
            self.smallestEnergy = atoms.get_potential_energy()
            key_value_pairs = self.db.get(name=self.name).key_value_pairs
            del self.db[self.runID]
            self.runID = self.db.write( atoms, key_value_pairs=key_value_pairs )

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

def main( argv ):
    relaxCell=False
    system = "AlMg"
    runID = int(argv[0])
    print ("Running job: %d"%(runID))
    db_paths = ["/home/ntnu/davidkl/GPAWTutorial/CE/ce_hydrostatic.db", "ce_hydrostatic.db","/home/davidkl/GPAWTutorial/CE/ce_hydrostatic.db"]
    for path in db_paths:
        if ( os.path.isfile(path) ):
            db_name = path
            break
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
    if ( system == "AlMg" and new_run==False ):
        atoms = change_cell_composition_AlMg(atoms)

    convergence = {
        "density":1E-4,
        "eigenstates":4E-8
    }
    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(4,4,4), nbands="120%", convergence=convergence )
    atoms.set_calculator( calc )

    logfile = "ceTest%d.log"%(runID)
    traj = "ceTest%d.traj"%(runID)
    trajObj = Trajectory(traj, 'w', atoms )

    storeBest = SaveToDB(db_name,runID,name)

    try:
        if ( relaxCell ):
            precon = Exp(mu=1.0,mu_c=1.0)
            uf = UnitCellFilter( atoms, hydrostatic_strain=True )
            relaxer = PreconLBFGS( uf, logfile=logfile, use_armijo=True, precon=precon )
        else:
            relaxer = BFGS( atoms, logfile=logfile )
        relaxer.attach( trajObj )
        relaxer.attach( storeBest, interval=1, atoms=atoms )
        if ( relaxCell ):
            relaxer.run( fmax=0.025, smax=0.003 )
        else:
            relaxer.run( fmax=0.025 )
        energy = atoms.get_potential_energy()
        db.update( storeBest.runID, converged=True )
        print ("Energy: %.2E eV/atom"%(energy/len(atoms)) )
        print ("Preconditioner parameters")
        print ("Mu:", precon.mu)
        print ("Mu_c:", precon.mu_c)
    except Exception as exc:
        print (exc)

if __name__ == "__main__":
    main( sys.argv[1:] )
