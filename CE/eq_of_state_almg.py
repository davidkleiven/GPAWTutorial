import gpaw as gp
from ase.build import bulk
from ase.db import connect
import numpy as np
from ase.io import read
import sys
import os

db_names = ["bulk_modulus_fcc.db","/home/ntnu/davidkl/GPAWTutorial/CE/bulk_modulus_fcc.db"]

db_name = ""
for name in db_names:
    if ( os.path.isfile(name) ):
        db_name = name
        break

def insert_structures():
    db = connect( db_name )
    a = [3.9,4.0,4.1]
    atoms = read( "data/al3mg_bulkmod.xyz" )
    ref_cell = atoms.get_cell()
    a0 = 4.05
    for i in range( len(a) ):
        new_cell = ref_cell*a[i]/a0
        atoms.set_cell( new_cell, scale_atoms=True )
        db.write(atoms)

def main( runID ):
    db = connect( db_name )
    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(8,8,8), nbands=-50 )
    atoms = db.get_atoms(id=runID)
    atoms.set_calculator( calc )
    energy = atoms.get_potential_energy()
    del db[runID]
    db.write( atoms )

if __name__ == "__main__":
    insert_structures()
    #main( sys.argv[1] )
