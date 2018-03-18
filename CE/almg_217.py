import sys
from ase.spacegroup import crystal
from ase.visualize import view
from ase.ce.settings_bulk import BulkSpacegroup
from ase.ce.newStruct import GenerateStructures
from almg_bcc_ce import insert_specific_structure
from ase.io import read

def get_atoms():
    # https://materials.springer.com/isp/crystallographic/docs/sd_0453869
    a = 10.553
    b = 10.553
    c = 10.553
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a,b,c,alpha,beta,gamma]
    symbols = ["Mg","Mg","Mg","Al"]
    #symbols = ["Al","Al","Al","Al"]
    basis = [(0,0,0),(0.324,0.324,0.324),(0.3582,0.3582,0.0393),(0.0954,0.0954,0.2725)]
    atoms = crystal( symbols, spacegroup=217, cellpar=cellpar, basis=basis)
    return atoms, cellpar,symbols,basis

def get_atoms_pure():
    # https://materials.springer.com/isp/crystallographic/docs/sd_0453869
    a = 9.993
    b = 9.993
    c = 9.993
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a,b,c,alpha,beta,gamma]
    symbols = ["Mg","Mg","Mg","Al"]
    #symbols = ["Al","Al","Al","Al"]
    basis = [(0,0,0),(0.332,0.332,0.332),(0.362,0.362,0.051),(0.0949,0.0949,0.2864)]
    atoms = crystal( symbols, spacegroup=217, cellpar=cellpar, basis=basis)
    return atoms, cellpar,symbols,basis

db_name = "almg_217_test.db"
def main( argv ):
    option = argv[0]
    atoms, cellpar, symbols, basis = get_atoms_pure()
    conc_args = {
        "conc_ratio_min_1":[[1,0],[1,0],[1,0],[1,0]],
        "conc_ratio_max_1":[[0,1],[0,1],[0,1],[0,1]]
    }
    conc_args = {
        "conc_ratio_min_1":[[1,0],[1,0]],
        "conc_ratio_max_1":[[0,1],[0,1]]
    }
    basis_elements = [["Al","Mg"],["Al","Mg"],["Al","Mg"],["Al","Mg"]]
    bs = BulkSpacegroup( basis_elements=basis_elements, basis=basis, spacegroup=217, cellpar=cellpar, conc_args=conc_args,
    max_cluster_size=4, db_name=db_name, size=[1,1,1], grouped_basis=[[0,1,2,3]] )
    print (bs.cluster_indx[2])
    exit()

    struct_gen = GenerateStructures( bs, struct_per_gen=10 )

    if ( option == "newstruct" ):
        struct_gen.generate_probe_structure()
    elif ( option == "insert" ):
        fname = argv[1]
        atoms = read( fname )
        insert_specific_structure( bs, struct_gen, atoms )

if __name__ == "__main__":
    main( sys.argv[1:] )
