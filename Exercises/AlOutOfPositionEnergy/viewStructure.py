import sys
from ase.visualize import view
import sqlite3 as sq
import ase.db

def main( argv ):
    runID = int( argv[0] )
    db_name = "aloutofpos.db"
    con = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "SELECT systemID FROM simpar WHERE ID=?", (runID,) )
    sysID = cur.fetchone()[0]
    con.close()
    asedb = ase.db.connect( db_name )
    atoms = asedb.get_atoms( selection=sysID )
    view( atoms, viewer="Avogadro" )


if __name__ == "__main__":
    main( sys.argv[1:] )
