from ase import build
import gpaw as gp
from ase.optimize.precon import PreconLBFGS
from ase.io.trajectory import Trajectory
from ase.constraints import UnitCellFilter
import numpy as np
import random as rnd
import copy as cp
import pickle as pck
from ase import parallel

def main():
    atoms = build.bulk( "Al" )
    atoms = atoms*(2,2,2)
    print (len(atoms))

    nRuns = 10
    optimizerFname = "optimizer.pck"
    for i in range(nRuns):
        nMgAtoms = np.random.randint(0,len(atoms)/2)

        # Insert Mg atoms
        system = cp.copy(atoms)
        if ( parallel.rank == 0 ):
            for j in range(nMgAtoms):
                system[i].symbol = "Mg"

            # Shuffle the list
            for j in range(10*len(system)):
                first = np.random.randint(0,len(system))
                second = np.random.randint(0,len(system))
                symb1 = system[first].symbol
                system[first].symbol = system[second].symbol
                system[second].symbol = symb1
        system = parallel.broadcast( system )

        # Initialize the calculator
        calc = gp.GPAW( mode=gp.PW(400), xc="PBE", kpts=(4,4,4), nbands=-10 )
        system.set_calculator( calc )

        traj = Trajectory( "trajectoryResuse.traj", 'w', atoms )

        if ( i == 0 ):
            relaxer = PreconLBFGS( UnitCellFilter( system ), logfile="resuse.log" )
        else:
            relaxer = None
            if ( parallel.rank == 0 ):
                with open( optimizerFname, 'r' ) as infile:
                    relaxer = pck.load( infile )
            parallel.barrier()
            relaxer = parallel.broadcast( relaxer )

            # Change the system, but re use the preconditioner
            relaxer.atoms = UnitCellFilter( system )
        relaxer.attach( traj )

        relaxer.run( fmax=0.05 )
        print (relaxer.iteration)
        if ( parallel.rank == 0 ):
            with open( optimizerFname, 'w' ) as outfile:
                pck.dump( relaxer, outfile )
        parallel.barrier()

if __name__ == "__main__":
    main()
