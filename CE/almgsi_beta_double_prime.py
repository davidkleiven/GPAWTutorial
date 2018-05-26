import sys
from ase.spacegroup import crystal
from ase.visualize import view
from ase.ce import BulkSpacegroup
from ase.ce.newStruct import GenerateStructures

def get_atoms():
    # https://materials.springer.com/isp/crystallographic/docs/sd_1123566
    a = 15.16
    b = 4.05
    c = 6.74
    alpha = 90
    beta = 105.3
    gamma = 90
    cellpar = [a,b,c,alpha,beta,gamma]
    symbols = ["Si","Si","Si","Mg","Mg","Mg"]
    basis = [(0.0565,0.0,0.649),(0.1885,0.0,0.224),(0.2171,0.0,0.617),(0.3459,0.0,0.089),(0.57,0.0,0.348),(0,0,0)]
    atoms = crystal( symbols, spacegroup=12, cellpar=cellpar, basis=basis)
    return atoms, cellpar,symbols,basis

db_name = "almgsi_beta_double_prime.db"
def main( argv ):
    option = argv[0]
    atoms, cellpar, symbols, basis = get_atoms()
    conc_args = {
        "conc_ratio_min_1":[[1,0],[1,0],[1,0],[1,0]],
        "conc_ratio_max_1":[[0,1],[0,1],[0,1],[0,1]]
    }
    print (len(atoms))
    view(atoms)
    exit()
    """
    basis_elements = [["Al","Mg"],["Al","Mg"],["Al","Mg"],["Al","Mg"]]
    bs = BulkSpacegroup( basis_elements=basis_elements, basis=basis, spacegroup=12, cellpar=cellpar, conc_args=conc_args,
    max_cluster_size=4, db_name=db_name, size=[1,1,1] )

    struct_gen = GenerateStructures( bs, struct_per_gen=10 )

    if ( option == "newstruct" ):
        struct_gen.generate_probe_structure()
    """

if __name__ == "__main__":
    main( sys.argv[1:] )
