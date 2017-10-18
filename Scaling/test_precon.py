import sys
import gpaw as gp
from ase.optimize.precon import PreconLBFGS
from ase.optimize.precon.precon import Exp
from ase.io.trajectory import Trajectory
from ase.build import bulk

def main( argv ):
    n_mg = int(argv[0])
    atoms = bulk("Al")
    atoms = atoms*(3,3,3)
    for i in range(n_mg):
        atoms[i].symbol = "Mg"

    atoms.rattle()

    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(6,6,6), nbands="120%" )
    atoms.set_calculator( calc )

    logfile = "preconTest%d.log"%(n_mg)
    traj = "preconTest%d.traj"%(n_mg)
    trajObj = Trajectory(traj, 'w', atoms )

    relaxer = PreconLBFGS( atoms, logfile=logfile, use_armijo=True )
    relaxer.attach( trajObj )
    relaxer.run( fmax=0.001 )
    print ("Mu: %.2E"%(relaxer.precon.mu))

if __name__ == "__main__":
    main( sys.argv[1:] )
