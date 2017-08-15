import sys
from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
import ase.db
import sqlite3 as sq
from ase import build

def main( argv ):
    if ( len(argv) != 1 ):
        print ("Usage: python almg.py paramID")
        return

    db_name = "/home/ntnu/davidkl/GPAWTutorial/Exercises/AlMg/AlMg.db"
    db = ase.db.connect( db_name )
    runID = argv[0]

    # Read parameters from the database
    con  = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "SELECT hspacing,relax,atomID FROM runs WHERE ID=?", runID )
    params = cur.fetchall()[0]
    con.close()

    save_pov = False
    run_sim = True
    h_spacing = params[0]
    relax = params[1]
    atom_row_id = params[2]

    # Generate super cell
    NatomsX = 2
    NatomsY = 2
    NatomsZ = 4

    # Lattice parameter
    a = 4.05

    if ( atom_row_id < 0 ):
        # Target primitive cell
        atoms = bulk( "Al", crystalstructure="fcc" )

        # Create a supercell consisting of 32 atoms
        P = build.find_optimal_cell_shape( atoms, 32, "sc" )
        atoms = build.make_supercell( atoms, P )
        print (len(atoms))

        print ("Number of atoms: %d"%( len(atoms)) )

        # Replace some atoms with Mg atoms
        n_mg_atoms = int( 0.2*len(atoms) )

        for i in range(n_mg_atoms):
            atoms[i].set( "symbol", "Mg" )
    else:
        # Read atoms from database
        atoms = db.get_atoms( selection=atom_row_id )

    if ( save_pov ):
        from ase.io import write
        write( "Al.pov", atoms*(3,3,1), rotation="-10z,-70x" )

    if ( run_sim ):
        from gpaw import GPAW
        calc = GPAW( h=h_spacing, xc="PBE" )
        atoms.set_calculator( calc )

        if ( relax ):
            from ase.optimize import QuasiNewton
            relaxer = QuasiNewton( atoms, logfile="relaxation.log" )
            relaxer.run( fmax=0.05 )
        else:
            energy = atoms.get_potential_energy()
            print ("Energy %.2f eV/atom"%(energy) )
        lastID = db.write( atoms, relaxed=True )

        # Update the database
        con = sq.connect( db_name )
        cur = con.cursor()
        cur.execute( "UPDATE runs SET status='finished',resultID=? WHERE ID=?", (lastID, runID) )
        con.commit()
        con.close()

if ( __name__ == "__main__" ):
    main( sys.argv[1:] )
