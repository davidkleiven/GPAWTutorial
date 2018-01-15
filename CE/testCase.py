import ase.db
from ase.build import bulk
import numpy as np
from ase.visualize import view

def main():
    atoms = bulk("Al",crystalstructure="fcc", a=4.6, cubic=False)
    atoms = atoms*(2,2,2)
    print (len(atoms))
    db = ase.db.connect("test_db.db")
    tid = db.write(atoms, name="testCase%d"%(np.random.randint(0,1000000000)),started=False,queued=False)
    print ("Test ID: %d"%(tid))

if __name__ == "__main__":
    main()
