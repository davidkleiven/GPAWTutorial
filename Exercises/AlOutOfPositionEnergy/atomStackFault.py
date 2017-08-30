from __future__ import print_function
import sys
import os
import gpaw as gp
import sys
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
import ase.db
import sqlite3 as sq
from ase.constraints import StrainFilter
from ase.io.trajectory import Trajectory
from ase import build
import numpy as np
from ase.optimize import BFGS

def moveAtoms( atoms, n_atoms_to_shift, alat=4.05 ):
    """
    To move an atom from A->B or B->C it has to be translated
    a distance r*(1,1/sqrt(3) ) in the plane orthogonal to the (1,1,1) plane
    r is the radius of the spheres
    """
    r = alat/(2.0*np.sqrt(2.0)) # For FCC lattice
    xhat = np.array( [1,-1,0] )/np.sqrt(2)
    yhat = np.array( [-1,1,0] )/np.sqrt(2)

    # Translation vec normal t0 (1,1,1)
    d = np.array( [1.0,1/np.sqrt(3),0] )*r
    translation = np.zeros(3)
    translation += d[0]*xhat
    translation += d[1]*yhat

    for i in range(n_atoms_to_shift):
        atoms[i].x += translation[0]
        atoms[i].y += translation[1]
    return atoms

def main( argv ):
    runID = int( argv[0] )

    params = {
        "cutoff":200,
        "kpts":1,
        "n_atoms_to_shift":0,
        "nbands":-1,
        "relax":False,
        "gamma":False
    }

    db_name = "none"
    dbPaths = [
        "/home/ntnu/davidkl/GPAWTutorial/Exercises/AlOutOfPositionEnergy/aloutofpos.db",
        "aloutofpos.db"
    ]

    # Find the correct database
    for name in dbPaths:
        if ( os.path.isfile(name) ):
            db_name = name
            break

    if ( db_name == "none" ):
        print ("Could not find database")
        return

    # Read parameters from database
    con = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "SELECT cutoff,kpts,n_atoms_to_shift,nbands,relax,gamma FROM simpar WHERE ID=?", (runID,) )
    dbparams = cur.fetchall()[0]
    con.close()

    # Transfer the parameters to the params dictionary
    params["cutoff"] = dbparams[0]
    params["kpts"] = dbparams[1]
    params["n_atoms_to_shift"] = dbparams[2]
    params["nbands"] = dbparams[3]
    params["relax"] = dbparams[4]
    params["gamma"] = dbparams[5]

    if ( params["gamma"] ):
        kpts = {"size":(params["kpts"],params["kpts"],params["kpts"]), "gamma":True}
    else:
        kpts = (params["kpts"],params["kpts"],params["kpts"])
    # Initialize the calculator
    calc = gp.GPAW( mode=gp.PW(params["cutoff"]), xc="PBE", nbands=params["nbands"], kpts=kpts )

    # Initialize the atoms
    aluminum = build.bulk( "Al", crystalstructure="fcc" )
    P = build.find_optimal_cell_shape_pure_python( aluminum.cell, 32, "sc" )
    aluminum = build.make_supercell( aluminum, P )

    aluminum = moveAtoms( aluminum, params["n_atoms_to_shift"], alat=4.05 )
    aluminum.set_calculator( calc )
    print (aluminum.get_positions())
    exit()
    if ( params["relax"] ):
        logfile = "logilfe%d.log"%(runID)
        trajfile = "optTrajectory%d.traj"%(runID)

        traj = Trajectory( trajfile, 'w', aluminum )
        # Optimize cell
        strain = StrainFilter( aluminum )
        relaxer = BFGS( strain, logfile=logfile )
        relaxer.attach( traj )
        relaxer.run( fmax=0.05 )

        # Optimize internal positions
        relaxer = BFGS( aluminum, logfile=logfile )
        relaxer.attach( traj )
        relaxer.run( fmax=0.05 )

    energy = aluminum.get_potential_energy()

    # Add results to the database
    asedb = ase.db.connect( db_name )
    lastID = asedb.write( aluminum, relaxed=True )

    # Update the parameters in the database
    con = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "UPDATE simpar SET status=?,systemID=? WHERE ID=?", ("finished",lastID,runID) )
    con.commit()
    con.close()

if __name__ == "__main__":
    main( sys.argv[1:] )
