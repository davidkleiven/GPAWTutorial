import sys
from ase.db import connect
from ase.visualize import view

def main( uid ):
    db_name = "ce_hydrostatic.db"
    db = connect( db_name )
    atoms = db.get_atoms( id=uid )
    print ( "Chemical formula {}".format(atoms.get_chemical_formula()) )
    view(atoms)

if __name__ == "__main__":
    main( sys.argv[1] )
