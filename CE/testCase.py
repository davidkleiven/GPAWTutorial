import ase.db
from ase.build import bulk
import numpy as np
from ase.visualize import view

def main():
    atoms = bulk("Al",crystalstructure="fcc", a=4.6, cubic=True)
    atoms = atoms*(2,2,2)
    print (len(atoms))
    view(atoms, viewer="Avogadro")
    exit()
    db = ase.db.connect("ceTest.db")
    tid = db.write(atoms, name="testCase%d"%(np.random.randint(0,1000000000)))
    print ("Test ID: %d"%(tid))

if __name__ == "__main__":
    main()
