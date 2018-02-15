import sys
from ase.db import connect
import gpaw as gp

#db_name = "almg_inter_conv.db"
db_name = "/home/davidkl/GPAWTutorial/CE/almg_inter_conv.db"
def main( runID ):
    db = connect( db_name )
    atoms = db.get_atoms( id=runID )
    row = db.get(id=runID)
    N = row.kpt
    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(N,N,N), nbands=-50)
    atoms.set_calculator( calc )
    energy = atoms.get_potential_energy()
    db.update( runID, tot_energy=energy)

if __name__ == "__main__":
    main( sys.argv[1])
