import sys
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from ase.io.trajectory import Trajectory
from ase.build import bulk
import os
from gpaw.utilities.tools import cutoff2gridspacing

def main( argv ):
    atoms = bulk("Al")
    atoms = atoms*(4,4,4)
    mode = "fd"

    e_cut = 500
    if ( mode == "fd" ):
        h = cutoff2gridspacing(e_cut)
        calc = gp.GPAW(mode="fd", h=h, xc="PBE", kpts=(6,6,6), nbands=-10 )
    else:
        calc = gp.GPAW( mode=gp.PW(e_cut), xc="PBE", kpts=(6,6,6), nbands=-10 )
    atoms.set_calculator( calc )

    energy = atoms.get_potential_energy()
    print ("Energy: %.2E"%(energy))

if __name__ == "__main__":
    main( sys.argv[1:] )
