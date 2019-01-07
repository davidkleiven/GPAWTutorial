from ase.db import connect
from ase.visualize import view
import json

db = connect("pre_beta_simple_cubic.db")
REJECTED_NAME_FILE="data/rejected_names_based_on_angles.json"

def get_names():
    names = []
    for row in db.select(converged=1):
        names.append(row.name)
    return names

def filter_angles():
    names = get_names()
    ok_names = []
    rejected_names = []
    max_dev = 10.0
    rejected_struct = []
    for name in names:
        reject = False
        for row in db.select(name=name):
            atoms = row.toatoms()
            angles = atoms.get_cell_lengths_and_angles()[3:]
            if min(angles) < 90 - max_dev or max(angles) > 90 + max_dev:
                reject = True
                rejected_struct.append(atoms)
        
        if reject:
            rejected_names.append(name)
        else:
            ok_names.append(name)

    print("Num. rejected: {}".format(len(rejected_names)))
    print(rejected_names)
    #view(rejected_struct)
    data = {"max_angle_changes": max_dev, "names": rejected_names}
    with open(REJECTED_NAME_FILE, 'w') as outfile:
        json.dump(data, outfile)
    print("Rejected names written to {}".format(REJECTED_NAME_FILE))

def main():
    names = get_names()    
    for name in names:
        print(name)
        atoms = []
        for row in db.select(name=name):
            atoms.append(row.toatoms())
        view(atoms)
        msg = input("Press n to continue. e will abort: ")
        if msg != "n":
            break

if __name__ == "__main__":
    #main()
    filter_angles()