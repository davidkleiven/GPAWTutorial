from __future__ import print_function
import ase
from ase.optimize import QuasiNewton
import gpaw as gp

def main():
    molecule, calc = gp.restart( "H2.gpw", txt="H2-relaxed.txt" )
    print ("Getting potential energy")
    e2 = molecule.get_potential_energy()
    d0 = molecule.get_distance( 0, 1 )

    # Find the minimum energy by Quasi-Newton
    relax = QuasiNewton( molecule, logfile="qn.log" )
    relax.run( fmax=0.05 )
    d1 = molecule.get_distance( 0, 1 )

    print ("Original bondlength: %.2f"%(d0))
    print ("Optimized bondlength: %.2f"%(d1))

if __name__ == "__main__":
    main()
