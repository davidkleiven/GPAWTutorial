import sys
from ase.io import read, write
from ase.visualize import view
from ase.optimize.precon import PreconLBFGS, Exp
from ase.calculators.emt import EMT
import gpaw as gp
from ase.build import bulk
from ase.io.trajectory import Trajectory

def main( argv ):
    fname = argv[0]
    atoms = read( fname )
    if( int(argv[1]) == 1 ):
        variable_cell = True
    else:
        variable_cell=False
    #atoms = bulk("Al")

    calc = gp.GPAW( mode=gp.PW(500), kpts=(4,4,4), xc="PBE", nbands="120%" )
    atoms.set_calculator( calc )
    prc = Exp(mu=1.0,mu_c=1.0)
    relaxer = PreconLBFGS( atoms, logfile="al12mg17.log", precon=prc, use_armijo=True, variable_cell=variable_cell )
    trajObj = Trajectory("al3mg2.traj", 'w', atoms )
    relaxer.attach( trajObj )
    if ( variable_cell ):
        relaxer.run( fmax=0.025, smax=0.003 )
    else:
        relaxer.run( fmax=0.025 )
    outfname = fname.split("/")[-1]
    outfname = outfname.split(".")[0]
    outfname += "_relaxed.xyz"
    print ( "Energy: {}".format(atoms.get_potential_energy() ) )
    write( outfname, atoms )

if __name__ == "__main__":
    main( sys.argv[1] )
