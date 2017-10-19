import sys
import gpaw as gp
from ase.build import bulk

def main( argv ):
    atoms = bulk("Al")
    kpt = int(argv[0])
    n_atoms = int(argv[1])
    h = float(argv[0])
    atoms = atoms*(n_atoms,n_atoms,n_atoms)
    mode = "fd"

    calc = gp.GPAW(mode="fd", h=0.25, xc="PBE", kpts=(kpt,kpt,kpt), nbands="120%" )
    atoms.set_calculator( calc )

    energy = atoms.get_potential_energy()
    print ("Energy: %.2E"%(energy))

if __name__ == "__main__":
    main( sys.argv[1:] )
