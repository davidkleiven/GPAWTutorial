from ase import Atoms
from ase.io import write
from gpaw import GPAW
import h5py as hf

def orbitals():
    atom = Atoms("O", cell=[6,6,6], pbc=False )
    atom.center()

    calc = GPAW( h=0.2, hund=True, txt="O.txt" )
    atom.set_calculator( calc )
    atom.get_potential_energy()

    calc.write( "data/O.gpw", mode="all" )

    # Cube files for orbitals
    h5file = hf.File( "data/wavefunc.h5", 'w' )
    for spin in [0,1]:
        for n in range(calc.get_number_of_bands() ):
            wf = calc.get_pseudo_wave_function( band=n, spin=spin )
            h5file.create_dataset( "wf%d_%d"%(spin,n), data=wf )
    h5file.close()


def main():
    orbitals()

if __name__ == "__main__":
    main()
