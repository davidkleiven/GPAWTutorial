import sys
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.build import bulk
import os

def main( argv ):
    atoms = bulk("Al")
    atoms = atoms*(4,4,4)

    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(6,6,6), nbands=-10 )
    atoms.set_calculator( calc )

    energy = atoms.get_potential_energy()
    print ("Energy: %.2E"%(energy))

if __name__ == "__main__":
    main( sys.argv[1:] )
