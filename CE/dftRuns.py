import sys
import gpaw as gp
import ase.db
from ase.optimize.precon import PreconLBFGS
from ase.constraints import UnitCellFilter
from ase.io.trajectory import Trajectory

def main( argv ):
    runID = int(sys.argv[0])
    db_name = "/home/ntnu/davidkl/Documents/GPAWTutorials/ceTest.db"
    db = ase.db.connect( db_name )

    # Update the databse
    db.update( runID, started=True )
    db.update( runID, collapsed=True )
    atoms = db.get_atoms(id=runID)

    cnvg = {
        "density":1E-2,
        "eigenstates":5E-3
    }
    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(6,6,6), nbands=-10 )
    calc.set( convergence=cnvg )
    atoms.set_calculator( calc )

    logfile = "ceTest%d.log"%(runID)
    traj = "ceTest%d.traj"%(runID)

    uf = UnitCellFilter(atoms)
    relaxer = PreconLBFGS( uf, logfile=logfile )
    trajObj = Trajectory(traj)
    relaxer.attach( traj )
    relaxer.run( fmax=0.05 )

    energy = atoms.get_potential_energy()
    db.update( runID, energy=energy )
    db.update( runID, positions=atoms.get_positions() )
    db.update( runID, collapsed=False )

if __name__ == "__main__":
    main( sys.argv[1:] )





if __name__ == "__main__":
    main( sys.argv[1:] )
