from ase import build
import gpaw as gp

def main():
    atoms = build.bulk( "Al" )
    atoms = atoms*(8,4,4)
    nMg = int( 0.2*len(atoms) )
    for i in range(nMg):
        atoms[i].symbol = "Mg"

    calc = gp.GPAW( mode=gp.PW(400), xc="PBE", nbands=-10, kpts=(4,4,4) )
    atoms.set_calculator( calc )
    energy = atoms.get_potential_energy()/len(atoms)

if __name__ == "__main__":
    main()
