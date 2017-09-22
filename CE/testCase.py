import ase.db
from ase.build import bulk
import numpy as np

def main():
    atoms = bulk("Al",crystalstructure="fcc", a=4.6)
    atoms = atoms*(2,1,1)
    db = ase.db.connect("ceTest.db")
    tid = db.write(atoms, name="testCase%d"%(np.random.randint(0,1000000000)))
    print ("Test ID: %d"%(tid))

if __name__ == "__main__":
    main()
