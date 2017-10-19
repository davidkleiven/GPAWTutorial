import sys
import gpaw as gp
from ase.optimize.precon import PreconLBFGS
from ase.optimize import BFGS
from ase.optimize.precon.precon import Exp
from ase.io.trajectory import Trajectory
from ase.build import bulk

def main( argv ):
    n_mg = int(argv[0])
    atoms = bulk("Al")
    atoms = atoms*(4,4,4)
    for i in range(n_mg):
        atoms[i].symbol = "Mg"

    atoms.rattle( stdev=0.005 )

    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(4,4,4), nbands="120%" )
    atoms.set_calculator( calc )

    logfile = "preconTest%d.log"%(n_mg)
    traj = "preconTest%d.traj"%(n_mg)
    trajObj = Trajectory(traj, 'w', atoms )

    relaxer = BFGS( atoms, logfile=logfile )
    relaxer.attach( trajObj )
    try:
        relaxer.run( fmax=0.05 )
    except:
        pass

if __name__ == "__main__":
    main( sys.argv[1:] )
