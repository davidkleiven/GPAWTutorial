from gpaw import GPAW
from ase.build import bulk

def main():
    h = 0.180.25
    atoms = bulk("Al")*(6, 6, 6)
    calc = GPAW(h=h, xc="PBE", kpts={"density": 1.37}, nbands="120%")
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    calc.write("gpaw_test.gpw")
    
if __name__ == "__main__":
    main()