import gpaw as gp
import ase.db
from ase.build import bulk
import random
import os
import sys


def db_name():
    candidates = ["almgsi_conv.db","/home/davidkl/GPAWTutorial/CE/almgsi_conv.db"]
    for cand in candidates:
        if ( os.path.exists(cand) ):
            return cand
    return candidates[0]

def prepare_db():
    atoms = bulk("Al", a=4.15)
    atoms = atoms*(4,4,4)
    for i in range(0,42):
        if ( i < 21 ):
            atoms[i].symbol = "Si"
        else:
            atoms[i].symbol = "Mg"

    symbols = [atom.symbol for atom in atoms]
    random.shuffle(symbols)
    for symb,atom in zip(symbols,atoms):
        atom.symbol = symb

    db = ase.db.connect( db_name() )
    for i in range(8):
        db.write(atoms)

def main( runID ):
    db = ase.db.connect( db_name() )
    atoms = db.get_atoms( id=runID )
    row = db.select( id=runID )
    n_kpt = row.n_kpt
    cutoff = row.cutoff

    calc = gp.GPAW( mode=gp.PW(cutoff), xc="PBE", kpts=(n_kpt,n_kpt,n_kpt), nbands="120%" )
    atoms.set_calculator( calc )
    energy = atoms.get_potential_energy()
    db.update( runID, trial_energy=energy )

if __name__ == "__main__":
    arg = sys.argv[1]
    if ( arg == "init" ):
        prepare_db()
    else:
        # Arg should be a run ID
        main(arg)
