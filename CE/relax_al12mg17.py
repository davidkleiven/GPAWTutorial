import sys
from ase.io import read, write
from ase.visualize import view
from ase.optimize.precon import PreconLBFGS, Exp
from ase.calculators.emt import EMT
import gpaw as gp
from ase.build import bulk

def main( fname ):
    atoms = read( fname )
    atoms = bulk("Al")

    calc = gp.GPAW( mode=gp.PW(500), kpts=(4,4,4), xc="PBE", nbands=-150 )
    atoms.set_calculator( calc )
    prc = Exp(mu=1.0,mu_c=1.0)
    relaxer = PreconLBFGS( atoms, logfile="al12mg17.log", precon=prc, use_armijo=True, variable_cell=True )
    relaxer.run( fmax=0.025, smax=0.003 )
    outfname = fname.split("/")[-1]
    outfname = outfname.split(".")[0]
    outfname += "_relaxed.xyz"
    print ( "Energy: {}".format(atoms.get_potential_energy() ) )
    write( outfname, atoms )

if __name__ == "__main__":
    main( sys.argv[1] )
