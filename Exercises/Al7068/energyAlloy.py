import al7068Factory as alfact
import gpaw as gp
from ase.io import write

def main():
    a = 4.05
    atoms = alfact.Al7068( symbol=("Al","Zn","Mg","Cu"), latticeconstant=a, pbc=True )
    #write( "al7068.pov", atoms, rotation="-10z,-70x" )

    atoms.set_cell( [6*a,6*a,6*a] )
    atoms.center()
    k=4
    #calc = gp.GPAW( xc="PBE", mode=gp.PW(300), kpts=(k,k,k) )
    calc = gp.GPAW( xc="PBE", h=0.15 )
    atoms.set_calculator( calc )
    energy = atoms.get_potential_energy()

if __name__ == "__main__":
    main()
