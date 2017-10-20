import sys
import gpaw as gp
from ase.optimize.precon import PreconLBFGS
from ase.optimize.precon.precon import Exp
from ase.io.trajectory import Trajectory
from ase.build import bulk
from ase.visualize import view

def main( argv ):
    atoms = bulk("Al")
    atoms = atoms*(6,6,6)
    for i in range(int(len(atoms)/5)):
        atoms[i].symbol = "Mg"


    atoms.rattle( stdev=0.005 )

    calc = gp.GPAW( mode="fd", h=0.2, xc="PBE", kpts=(2,2,2), nbands="120%" )
    atoms.set_calculator( calc )

    logfile = "relax250.log"
    traj = "relax250.traj"
    trajObj = Trajectory(traj, 'w', atoms )

    precon = Exp(mu=1)
    relaxer = PreconLBFGS( atoms, logfile=logfile, use_armijo=True, precon=precon )
    relaxer.attach( trajObj )
    try:
        relaxer.run( fmax=0.05 )
    except Exception as exc:
        print(exc)

if __name__ == "__main__":
    main( sys.argv[1:] )
