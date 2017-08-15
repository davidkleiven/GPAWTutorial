from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
import ase.db

def main():
    db_name = "AlMg.db"
    db = ase.db.connect( db_name )
    save_pov = False
    run_sim = True
    relax = True
    h_spacing = 0.25
    atom_row_id = -1
    # Generate super cell
    NatomsX = 2
    NatomsY = 2
    NatomsZ = 4

    # Lattice parameter
    a = 4.05

    if ( atom_row_id < 0 ):
        atoms = FaceCenteredCubic( directions=[[1,-1,0], [1,1,-2], [1,1,1]],
                size=(2,2,2), symbol="Al" )

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
        import gpaw import GPAW
        calc = GPAW( h=h_spacing, xc="PBE" )
        atoms.set_calculator( calc )

        if ( relax ):
            from ase.optimize import QuasiNewton
            relaxer = QuasiNewton( atoms, logfile="relaxation.log" )
            relaxer.run( fmax=0.05 )
        else:
            energy = atoms.get_potential_energy()
            print ("Energy %.2f eV/atom"%(energy) )
        db.write( atoms, relaxed=True )

if ( __name__ == "__main__" ):
    main()
