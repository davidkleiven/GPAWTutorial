import sys
sys.path.append("/home/davidkl/Documents/GPAWTutorial/CE")
sys.path.append("/home/davidkl/GPAWTutorial/CE")
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter,StrainFilter
from ase.io.trajectory import Trajectory
import os
import sqlite3 as sq
from ase.optimize.precon.precon import Exp
from ase.optimize.precon import PreconFIRE
from ase.optimize.sciopt import SciPyFminCG
from save_to_db import SaveToDB
def main( argv ):
    relax_mode = "volume" # both, volume, atoms
    system = "AlMg"
    runID = int(argv[0])
    print ("Running job: %d"%(runID))
    db_paths = ["/home/ntnu/davidkl/GPAWTutorial/CE/almgsi.db", "almgsi.db","/home/davidkl/GPAWTutorial/CE/almgsi.db"]
    for path in db_paths:
        if ( os.path.isfile(path) ):
            db_name = path
            break
    #db_name = "test_db.db"
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
    volume = atoms.get_volume()

    try:
        precon = Exp(mu=1.0,mu_c=1.0)
        fmax = 0.025
        smax = 0.003
        if ( relax_mode == "both" ):
            uf = UnitCellFilter( atoms, hydrostatic_strain=True )
            relaxer = PreconLBFGS( uf, logfile=logfile, use_armijo=True, precon=precon )
        elif ( relax_mode == "atoms" ):
            relaxer = SciPyFminCG( atoms, logfile=logfile )
        elif ( relax_mode == "volume" ):
            str_f = StrainFilter( atoms, mask=[1,1,1,0,0,0] )
            relaxer = SciPyFminCG( str_f, logfile=logfile )
            fmax=smax*volume

        relaxer.attach( trajObj )
        relaxer.attach( storeBest, interval=1, atoms=atoms )
        if ( relax_mode == "both" ):
            relaxer.run( fmax=fmax, smax=smax )
        else:
            relaxer.run( fmax=fmax )
        energy = atoms.get_potential_energy()

        if ( relax_mode == "atoms" ):
            db.update( storeBest.runID, converged_force=True )
        elif ( relax_mode == "volume" ):
            db.update( storeBest.runID, converged_stress=True )
        else:
            db.update( storeBest.runID, converged_stress=True, converged_force=True )

        row = db.get( id=runID )
        conv_force = row.get( "converged_force", default=False )
        conv_stress = row.get( "converged_stress", default=False )
        if ( conv_force and conv_stress ):
            db.update( storeBest.runID, converged=True )
    except Exception as exc:
        print (exc)

if __name__ == "__main__":
    main( sys.argv[1:] )
