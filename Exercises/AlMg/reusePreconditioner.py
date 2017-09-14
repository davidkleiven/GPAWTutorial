from ase import build
import gpaw as gp
from ase.optimize.precon import PreconLBFGS
from ase.io.trajectory import Trajectory
from ase.constraints import UnitCellFilter
import numpy as np
import random as rnd
import copy as cp
import pickle as pck

def main():
    atoms = build.bulk( "Al" )
    atoms = atoms*(2,2,2)

    nRuns = 10
    optimizerFname = "optimizer.pck"
    for i in range(nRuns):
        nMgAtoms = np.random.randint(0,len(atoms)/2)

        # Insert Mg atoms
        system = cp.copy(atoms)
        for j in range(nMgAtoms):
            system[i].symbol = "Mg"

        # Shuffle the list
        for j in range(10*len(system)):
            first = np.random.randint(0,len(system))
            second = np.random.randint(0,len(system))
            symb1 = system[first].symbol
            system[first].symbol = system[second].symbol
            system[second].symbol = symb1

        # Initialize the calculator
        calc = gp.GPAW( mode=gp.PW(400), kpts=(4,4,4), nbands=-10 )
        system.set_calculator( calc )

        if ( i == 0 ):
            relaxer = PreconLBFGS( system, trajectory="reuseTrajectory.traj", logfile="resuse.log" )
        else:
            with open( optimizerFname, 'r' ) as infile:
                relaxer = pck.load( infile )
            # Change the system, but re use the preconditioner
            relaxer.atoms = system

        relaxer.run( fmax=0.05 )
        print (relaxer.iteration)
        with open( optimizerFname, 'w' ) as outfile:
            pck.dump( relaxer, outfile )

if __name__ == "__main__":
    main()
