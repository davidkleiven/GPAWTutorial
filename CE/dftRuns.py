import sys
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS
from ase.constraints import UnitCellFilter
from ase.io.trajectory import Trajectory
import os
import sqlite3 as sq

def main( argv ):
    runID = int(argv[0])
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

    uf = UnitCellFilter(atoms)
    relaxer = PreconLBFGS( uf, logfile=logfile )
    trajObj = Trajectory(traj, 'w', atoms )
    relaxer.attach( trajObj )
    relaxer.run( fmax=0.05 )

    energy = atoms.get_potential_energy()
    db.update( runID, collapsed=False, converged=True )
    print ("Energy: %.2E eV/atom"%(energy/len(atoms)) )
    key_value_pairs = db.get(name=name).key_value_pairs
    del db[runID]
    newid = db.write( atoms, key_value_pairs=key_value_pairs )
    print ("New ID: %d"%(newid))

if __name__ == "__main__":
    main( sys.argv[1:] )
