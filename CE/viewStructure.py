import sys
import ase.db
from ase.visualize import view

def main( argv ):
    id = int( argv[0] )
    db = ase.db.connect("almgsi.db")
    atoms = db.get_atoms(id=id)
    view( atoms )
    atoms.write( "almgsi.xyz" )

if __name__ == "__main__":
    main( sys.argv[1:] )
