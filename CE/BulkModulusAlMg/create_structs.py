from ase.build import bulk
from ase.visualize import view

def main():
    atoms = bulk("Mg","fcc",a=4.05,cubic=True)
    view(atoms)

if __name__ == "__main__":
    main()
