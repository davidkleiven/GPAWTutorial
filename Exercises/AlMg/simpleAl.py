from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from gpaw import GPAW, PW
from ase import build

def main():
    atoms = build.bulk( "Al", crystalstructure="fcc" )
    calc = GPAW( mode=PW(400), xc="PBE", kpts=(8,8,8) )
    atoms.set_calculator( calc )
    energy = atoms.get_potential_energy()
    print (energy)

if __name__ == "__main__":
    main()
