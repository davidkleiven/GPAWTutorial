from ase.build import bulk, niggli_reduce
from ase.build import supercells as sc
from ase.visualize import view
from itertools import permutations
from ase.db import connect
from ase.io import read

def find_cell( tsize ):
    atoms = bulk("Al")
    trans = sc.find_optimal_cell_shape_pure_python( atoms.cell, tsize, "sc" )
    atoms = sc.make_supercell( atoms, trans )

    #atoms.set_cell(cell)
    view(atoms)

def generate_all_configurations( n_al, n_mg ):
    init_list = []
    for i in range(n_al):
        init_list.append("Al")
    for i in range(n_mg):
        init_list.append("Mg")

    all_combs = list( permutations( init_list, len(init_list) ) )

    # Filter the permuations
    filtered = []
    for perm in all_combs:
        already_exists = False
        for fperm in filtered:
            if is_equal_permutation( perm, fperm ):
                already_exists = True
                break
        if ( not already_exists ):
            filtered.append(perm)
    return filtered

def is_equal_permutation( list1, list2 ):
    for i1,i2 in zip(list1,list2):
        if ( i1 != i2 ):
            return False
    return True


def write_permutations_to_db( db_name, ref_xyz_file, symbols ):
    atoms = read( ref_xyz_file )
    niggli_reduce(atoms)
    db = connect( db_name )
    for symb_set in symbols:
        for i,symb in enumerate(symb_set):
            atoms[i].symbol = symb
        db.write( atoms )
    print ( "All intermetallic structures written to {}".format(db_name) )

def create_convergence_database( db_name ):
    atoms = read( "data/al3mg2_ref.xyz" )
    niggli_reduce(atoms)
    atoms[0].symbol = "Mg"
    atoms[1].symbol = "Mg"
    db = connect( db_name )
    kpt = [4,6,8,10,12,14]
    for N in kpt:
        db.write( atoms, kpt=N )
    print ( "Convergence study written to {}".format(db_name))

def main():
    #create_convergence_database( "almg_inter_conv.db" )
    #structs = generate_all_configurations( 3,2 )
    #write_permutations_to_db( "data/al3mg2_intermetallic.db", "data/al3mg2_ref.xyz", structs )
    pass

if __name__ == "__main__":
    main()
