import sys
import os
from ase.db import connect
from ase.io import write
from gpaw import restart

def main(argv):
    name = ""
    db_name = ""
    for arg in argv:
        if "--name=" in arg:
            name = arg.split("--name=")[1]
        elif "--db=" in arg:
            db_name = arg.split("--db=")[1]

    print("Retrieving data from {}".format(db_name))
    print("Selecting all structures with name {}".format(name))
    os.mkdir(name)
    db = connect(db_name)
    unknown_count = 0
    for row in db.select(name=name):
        print("Reading row {}".format(row.id))
        formula = row.formula
        attach_calc = row.get("calculator", "unknown") != "unknown"
        atoms = row.toatoms(attach_calculator=attach_calc)
        calc = atoms.get_calculator()
        struct_type = row.get("struct_type", "unknown")

        if struct_type == "unknown":
            struct_type += str(unknown_count)
            unknown_count += 1

        if calc is not None:
            try:
                atoms.get_potential_energy()
                calc.write("{}/{}.gpw".format(name, formula))
            except AttributeError as exc:
                print(exc)
        write("{}/{}_{}.xyz".format(name, formula, struct_type), atoms)

if __name__ == "__main__":
    main(sys.argv[1:])


