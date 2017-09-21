import sys
import ase.db
from ase.visualize import view

def main( argv ):
    id = int( argv[0] )
    db = ase.db.connect("ceTest.db")
    atoms = db.get_atoms(id=id)
    view( atoms, viewer="avogadro")

if __name__ == "__main__":
    main( sys.argv[1:] )
