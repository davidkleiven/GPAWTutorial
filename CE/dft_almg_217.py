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
from ase.optimize import BFGS
from ase.optimize.sciopt import SciPyFminCG
from ase.optimize import QuasiNewton
from save_to_db import SaveToDB
from ase.calculators.singlepoint import SinglePointCalculator
from atomtools.ase.save_restart import SaveRestartFiles

"""
class SaveRestartFiles(object):
    def __init__(self, calc, name):
        self.calc = calc
        self.name = name

    @staticmethod
    def restart_name(name):
        return fname = "calc_restart{}.gpw".format(name)

    def __call__(self):
        fname = SaveRestartFiles.restart_name(self.name)
        self.calc.write(fname)
"""

def main( argv ):
    relax_mode = "positions" # both, cell, positions
    system = "AlMg"
    runID = int(argv[0])
    nkpt = int(argv[1])

    single_point = False
    if ( len(argv) >= 3 ):
        single_point = (int(argv[2]) == 1 )
    print ("Running job: %d"%(runID))
    db_paths = ["/home/ntnu/davidkl/GPAWTutorial/CE/almg_217.db", "almg_217.db","/home/davidkl/GPAWTutorial/CE/almg_217.db"]
    for path in db_paths:
        if ( os.path.isfile(path) ):
            db_name = path
            break
    #db_name = "almgsi_test_db.db"
    db = ase.db.connect( db_name )
    name = db.get(id=runID).key_value_pairs["name"]
    new_run = not db.get( id=runID ).key_value_pairs["started"]
    
    # Update the databse
    db.update( runID, started=True, converged=False )
    db.update( runID, nkpt=nkpt )

    atoms = db.get_atoms(id=runID)

    if ( len(atoms) == 1 ):
        nbands = -10
    else:
        nbands = "120%"
    kpts = (nkpt,nkpt,nkpt)
    try:
        restart_name = SaveRestartFiles.restart_name( name )
        atoms, calc = gp.restart( restart_name )
    except:
        calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=kpts, nbands=nbands )
        atoms.set_calculator( calc )

    logfile = "almg_bcc%d.log"%(runID)
    traj = "almg_bcc%d.traj"%(runID)
    trajObj = Trajectory(traj, 'w', atoms )

    #storeBest = SaveToDB(db_name,runID,name,mode=relax_mode)
    save_calc = SaveRestartFiles(calc,name)
    volume = atoms.get_volume()

    try:
        precon = Exp(mu=1.0,mu_c=1.0)
        fmax = 0.025
        smax = 0.003
        if ( relax_mode == "both" ):
            relaxer = PreconLBFGS( atoms, logfile=logfile, use_armijo=True, precon=precon, variable_cell=True )
        elif ( relax_mode == "positions" ):
            #relaxer = SciPyFminCG( atoms, logfile=logfile )
            relaxer = BFGS( atoms, logfile=logfile )
        elif ( relax_mode == "cell" ):
            str_f = StrainFilter( atoms, mask=[1,1,1,0,0,0] )
            relaxer = BFGS( str_f, logfile=logfile )
            fmax=smax*volume

        relaxer.attach( trajObj )
        #relaxer.attach( storeBest, interval=1, atoms=atoms )
        relaxer.attach( save_calc, interval=1 )
        if ( not single_point ):
            if ( relax_mode == "both" ):
                relaxer.run( fmax=fmax, smax=smax )
            else:
                relaxer.run( fmax=fmax )
        energy = atoms.get_potential_energy()

        orig_atoms = db.get_atoms(runID)
        single_p_calc = SinglePointCalculator( orig_atoms, energy=energy )
        orig_atoms.set_calculator( single_p_calc )
        kvp = db.get(name=name).key_value_pairs
        del db[runID]
        newID = db.write( orig_atoms, key_value_pairs=kvp )

        if ( relax_mode == "positions" ):
            db.update( newID, converged_force=True )
        elif ( relax_mode == "cell" ):
            db.update( newID, converged_stress=True )
        else:
            db.update( newID, converged_stress=True, converged_force=True )

        db.update( newID, single_point=single_point )
        row = db.get( id=newID )
        conv_force = row.get( "converged_force", default=0 )
        conv_stress = row.get( "converged_stress", default=0 )
        if ( (conv_force==1) and (conv_stress==1) and (nkpt==4) ):
            db.update( newID, converged=True )
    except Exception as exc:
        print (exc)

if __name__ == "__main__":
    main( sys.argv[1:] )
