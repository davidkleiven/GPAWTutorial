import sys
from ase.db import connect
import gpaw as gp
from ase.optimize.precon import PreconLBFGS
from ase.optimize.precon.precon import Exp
from ase.constraints import UnitCellFilter

#db_name = "almg_inter_conv.db"
#db_name = "/home/davidkl/GPAWTutorial/CE/almg_inter_conv.db"
db_name = "/home/davidkl/GPAWTutorial/CE/al3mg2_intermetallic.db"
def main( runID ):
    db = connect( db_name )
    atoms = db.get_atoms( id=runID )
    N = 14
    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(N,N,N), nbands=-50, symmetry={'do_not_symmetrize_the_density': True} )
    atoms.set_calculator( calc )
    precon = Exp(mu=1.0,mu_c=1.0)
    uf = UnitCellFilter( atoms, hydrostatic_strain=True )
    logfile = "al3mg2%d.log"%(runID)
    relaxer = PreconLBFGS( uf, logfile=logfile, use_armijo=True, precon=precon )
    relaxer.run( fmax=0.025, smax=0.003 )
    energy = atoms.get_potential_energy()
    del db[db.get(id=runID)]
    db.write( atoms )

if __name__ == "__main__":
    main( sys.argv[1])
