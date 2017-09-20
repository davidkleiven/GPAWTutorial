import ase.db
from ase.build import bulk

def main():
    atoms = bulk("Al")
    db = ase.db.connect("ceTest.db")
    tid = db.write(atoms, name="testCase")
    print ("Test ID: %d"%(tid))

if __name__ == "__main__":
    main()
