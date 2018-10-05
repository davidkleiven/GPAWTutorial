import sys
from ase.ce import BulkCrystal, GenerateStructures

db_name = "almgsiX_fcc.db"
eci_fname = "data/eci_almgsix.json"

def main(argv):
    option = argv[0]
    conc_args = {
        "conc_ratio_min_1":[[64,0,0]],
        "conc_ratio_max_1":[[24,40,0]],
        "conc_ratio_min_2":[[64,0,0]],
        "conc_ratio_max_2":[[22,21,21]]
    }
    N = 6

    ceBulk = BulkCrystal(crystalstructure="fcc", a=4.05, size=[N,N,N], basis_elements=[["Al", "Mg", "Si", "X"]], \
    conc_args=conc_args, db_name=db_name, max_cluster_size=4, max_cluster_dia=[0.0, 0.0, 5.0, 4.1, 4.1])

    struc_generator = GenerateStructures( ceBulk, struct_per_gen=10 )
    if option == "reconfig_settings":
        ceBulk.reconfigure_settings()
    elif option == "insert":
        fname = argv[1]
        struc_generator.insert_structure(init_struct=fname)
    elif option == "random":
        atoms = get_random_structure()
        struc_generator.insert_structure(init_struct=atoms)


def get_random_structure():
    from ase.build import bulk
    from random import choice
    atoms = bulk("Al")*(6, 6, 6)
    symbs = ["Al", "Mg", "Si"]

    for atom in atoms:
        atom.symbol = choice(symbs)

    indx1 = choice(range(len(atoms)))
    indx2 = choice(range(len(atoms)))
    atoms[indx1].symbol = "X"
    atoms[indx2].symbol = "X"
    return atoms

if __name__ == "__main__":
    main(sys.argv[1:])
