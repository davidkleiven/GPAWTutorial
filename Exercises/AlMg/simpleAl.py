from ase import Atoms
from ase.lattice.cubic import FaceCenteredCubic
from gpaw import GPAW, PW
from ase import build
from ase.visualize import view

def viewSuperCell( atoms ):
        P = build.find_optimal_cell_shape_pure_python( atoms.cell, 32, "sc" )
        atoms = build.make_supercell( atoms, P )
        atoms[0].symbol="Mg"
        atoms[1].symbol = "Mg"
        atoms[2].symbol = "Mg"
        view( atoms, viewer="Avogadro" )

def main():
    atoms = build.bulk( "Al", crystalstructure="fcc" )
    calc = GPAW( mode=PW(400), xc="PBE", kpts=(8,8,8) )
    atoms.set_calculator( calc )
    energy = atoms.get_potential_energy()
    viewSuperCell(atoms)
    print (energy)

if __name__ == "__main__":
    main()
