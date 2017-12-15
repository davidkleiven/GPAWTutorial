import sys
sys.path.append("/home/davidkl/Documents/GPAWTutorial/CE")
sys.path.append("/home/davidkl/GPAWTutorial/CE")
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from ase.io.trajectory import Trajectory
import os
import sqlite3 as sq
from ase.optimize.precon.precon import Exp
from ase.optimize.precon import PreconFIRE
from ase.optimize.sciopt import SciPyFminCG
from save_to_db import SaveToDB
def main( argv ):
    relaxCell=False
    system = "AlMg"
    runID = int(argv[0])
    print ("Running job: %d"%(runID))
    db_paths = ["/home/ntnu/davidkl/GPAWTutorial/CE/almgsi_ce.db", "almgsi_ce.db.db","/home/davidkl/GPAWTutorial/CE/almgsi_ce.db.db"]
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

    calc = gp.GPAW( mode=gp.PW(600), xc="PBE", kpts=(4,4,4), nbands="120%" )
    atoms.set_calculator( calc )

    logfile = "almgsi%d.log"%(runID)
    traj = "almgsi%d.traj"%(runID)
    trajObj = Trajectory(traj, 'w', atoms )

    storeBest = SaveToDB(db_name,runID,name)

    try:
        precon = Exp(mu=1.0,mu_c=1.0)
        if ( relaxCell ):
            uf = UnitCellFilter( atoms, hydrostatic_strain=True )
            relaxer = PreconLBFGS( uf, logfile=logfile, use_armijo=True, precon=precon )
        else:
            relaxer = SciPyFminCG( atoms, logfile=logfile )
        relaxer.attach( trajObj )
        relaxer.attach( storeBest, interval=1, atoms=atoms )
        if ( relaxCell ):
            relaxer.run( fmax=0.025, smax=0.003 )
        else:
            relaxer.run( fmax=0.025 )
        energy = atoms.get_potential_energy()
        db.update( storeBest.runID, converged=True )
    except Exception as exc:
        print (exc)

if __name__ == "__main__":
    main( sys.argv[1:] )
