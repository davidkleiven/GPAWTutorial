import sys
import os
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
import ase.db
import sqlite3 as sq
from ase import build
from gpaw.poisson import PoissonSolver
import numpy as np
from ase.constraints import UnitCellFilter, StrainFilter
import random as rnd
from ase.io.trajectory import Trajectory

def main( argv ):
    if ( len(argv) > 2 ):
        print ("Usage: python almg.py paramID")
        return

    possible_dbs = {
        "vilje":"/home/ntnu/davidkl/GPAWTutorial/Exercises/AlMg/AlMg.db",
        "laptop":"AlMg.db"
    }

    # Find which database to use
    db_name = None
    for key in possible_dbs:
        if ( os.path.isfile(possible_dbs[key]) ):
            db_name = possible_dbs[key]
            break
    if ( db_name is None ):
        raise RuntimeError("Could not find database")

    #db_name = "/home/ntnu/davidkl/GPAWTutorial/Exercises/AlMg/AlMg.db"
    db = ase.db.connect( db_name )
    runID = int(argv[0])

    # Read parameters from the database
    con  = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "SELECT hspacing,relax,atomID,kpts,nbands,latticeConst,cutoff,traj,fmax,tags FROM runs WHERE ID=?", (runID,) )
    params = cur.fetchall()[0]
    con.close()

    save_pov = False
    run_sim = True
    swap_atoms = False
    h_spacing = params[0]
    relax = params[1]
    atom_row_id = params[2]
    Nkpts = params[3]
    nbands=params[4]
    cutoff = params[6]
    old_traj = params[7]
    fmax = params[8]
    tags = params[9]

    # Generate super cell
    if ( "test" in tags ):
        Natoms = 4
    else:
        Natoms = 32

    # Lattice parameter
    a = float( params[5] )

    if ( old_traj != "none" ):
        # Read the atoms from the trajectory file
        traj = Trajectory( old_traj )
        atoms = traj[-1]

    elif ( atom_row_id < 0 ):
        # Target primitive cell
        atoms = build.bulk( "Al", crystalstructure="fcc", a=a )

        # Create a supercell consisting of 32 atoms
        P = build.find_optimal_cell_shape_pure_python( atoms.cell, Natoms, "sc" )
        atoms = build.make_supercell( atoms, P )

        # Replace some atoms with Mg atoms
        n_mg_atoms = int( 0.2*len(atoms) )

        for i in range(n_mg_atoms):
            atoms[i].set( "symbol", "Mg" )
    else:
        # Read atoms from database
        atoms = db.get_atoms( selection=atom_row_id )

    if ( swap_atoms ):
        from ase.io import write
        for i in range(0, 2*len(atoms) ):
            first = np.random.randint(0,len(atoms))
            second = np.random.randint(0,len(atoms))
            firstSymbol = atoms[first].symbol
            atoms[first].symbol = atoms[second].symbol
            atoms[second].symbol = firstSymbol
        atoms.write( "almg_swap.xyz" )

    if ( save_pov ):
        from ase.io import write
        write( "Al.pov", atoms*(3,3,1), rotation="-10z,-70x" )
        return

    if ( run_sim ):
        from gpaw import GPAW, PW
        kpts = {"size":(Nkpts,Nkpts,Nkpts), "gamma":True} # Monkhorst pack

        if ( cutoff > 0 ):
            mode = PW(cutoff)
        else:
            mode = "fd"
        #calc = GPAW( mode="fd", h=h_spacing, xc="PBE", nbands=nbands, kpts=kpts, basis="dzp", poissonsolver=PoissonSolver(relax="GS", eps=1E-7) )
        calc = GPAW( mode=mode, xc="PBE", nbands=nbands, kpts=kpts )
        atoms.set_calculator( calc )

        if ( relax ):
            from ase.optimize import QuasiNewton, BFGS
            from ase.optimize.precon import PreconLBFGS

            uid = rnd.randint(0,10000000)
            # First relax only the unit cell
            logfile = "relaxation_%d.log"%( uid )
            trajfile = "trajectory_%d.traj"%( uid )

            strfilter = StrainFilter( atoms )
            relaxer = BFGS( strfilter, logfile=logfile, trajectory=trajfile )
            relaxer.run( fmax=fmax )

            # Relax atoms within the unit cell
            relaxer = PreconLBFGS( atoms, use_armijo=True, logfile=logfile, trajectory=trajfile )
            relaxer.run( fmax=fmax )

            # Optimize both simultaneously
            uf = UnitCellFilter( atoms )
            relaxer = BFGS( uf, logfile=logfile, trajectory=trajfile )
            relaxer.run( fmax=fmax )

        energy = atoms.get_potential_energy()
        print ("Energy %.2f eV/atom"%(energy) )
        lastID = db.write( atoms, relaxed=True )

        # Update the database
        con = sq.connect( db_name )
        cur = con.cursor()
        cur.execute( "UPDATE runs SET status='finished',resultID=?,logfile=?,traj=? WHERE ID=?", (lastID, runID, logfile,trajfile) )
        con.commit()
        con.close()

if ( __name__ == "__main__" ):
    main( sys.argv[1:] )
