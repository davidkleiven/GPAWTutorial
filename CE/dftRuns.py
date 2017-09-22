import sys
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from ase.io.trajectory import Trajectory
import os
import sqlite3 as sq

class SaveToDB(object):
    def __init__(self, db_name, runID, name):
        self.db = ase.db.connect( db_name )
        self.runID = runID
        self.name = name
        self.smallestEnergy = 1000.0

    def __call__(self, atoms=None):
        """
        Saves the current run to db if the energy is lower
        """
        if ( atoms is None ):
            return

        if ( atoms.get_potential_energy() < self.smallestEnergy ):
            self.smallestEnergy = atoms.get_potential_energy()
            key_value_pairs = self.db.get(name=self.name).key_value_pairs
            del self.db[self.runID]
            self.runID = self.db.write( atoms, key_value_pairs=key_value_pairs )

def main( argv ):
    runID = int(argv[0])
    print ("Running job: %d"%(runID))
    db_paths = ["/home/ntnu/davidkl/GPAWTutorial/CE/ceTest.db", "ceTest.db"]
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

    # Update the databse
    db.update( runID, started=True, converged=False, collapsed=True )
    atoms = db.get_atoms(id=runID)

    cnvg = {
        "density":1E-2,
        "eigenstates":5E-3,
        "bands":-10
    }
    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(6,6,6), nbands=-10 )
    calc.set( convergence=cnvg )
    atoms.set_calculator( calc )

    logfile = "ceTest%d.log"%(runID)
    traj = "ceTest%d.traj"%(runID)
    trajObj = Trajectory(traj, 'w', atoms )

    storeBest = SaveToDB(db_name,runID,name)

    uf = UnitCellFilter(atoms)
    relaxer = PreconLBFGS( uf, logfile=logfile, use_armijo=False )
    #relaxer = BFGS( atoms, logfile=logfile )
    relaxer.attach( trajObj )
    relaxer.attach( storeBest, interval=1, atoms=atoms )
    relaxer.run( fmax=0.05 )

    energy = atoms.get_potential_energy()

    db.update( storeBest.runID, collapsed=False, converged=True )
    print ("Energy: %.2E eV/atom"%(energy/len(atoms)) )

if __name__ == "__main__":
    main( sys.argv[1:] )
