from __future__ import print_function
import ase
import gpaw as gp
print (gp.__file__)

def main():
    unitcellSize = 10.0 # Size of unit cell in Angstrohm
    center = unitcellSize/2.0

    # Create Hydrogen atom located at the center of the unitcell
    atom = ase.Atoms( "H", positions=[(center,center,center)], magmoms=[0],
        cell=(center,center+0.001, center+0.002))

    # Initialize the GPAW calculator
    calc = gp.GPAW( mode=gp.PW(), xc="PBE", hund=True, eigensolver="rmm-diis",
    occupations=gp.FermiDirac(0.0, fixmagmom=True), txt="hydrogen.out")
    atom.set_calculator( calc )

    print ("Solving single atom...")
    e1 = atom.get_potential_energy()
    calc.write("H.gpw")

    # Simulate hydrogen molecule
    d = 0.74 # Bond length
    molecule = ase.Atoms( "H2", positions=([center-d/2.0,center,center],[center+d/2.0,center,center]),
        cell=(center,center,center))

    calc.set(txt="H.out")
    calc.set(hund=False)
    molecule.set_calculator( calc )
    print ("Solving hydrogen molecule...")
    e2 = molecule.get_potential_energy()

    # Test if it recalculates everytime
    for i in range(10):
        e2 = molecule.get_potential_energy()
        print (e2)
    calc.write("H2.gpw")

    print ( "Hydrogen atom energy: %.2f eV"%(e1) )
    print ( "Hydrogen molecule energy: %.2f eV"%(e2) )
    print ( "Atomization energy: %.2f eV"%(2*e1-e2) )

if __name__ == "__main__":
    main()
