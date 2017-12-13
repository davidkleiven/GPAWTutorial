import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
from ase.build import bulk
from ase.ce.settings import BulkCrystal
from ase.ce.newStruct import GenerateStructures

def main(argv):
    db_name = "almgsi.db"
    option = argv[0]
    conc_args = {
        "conc_ratio_min_1":[[1,0,0]],
        "conc_ratio_max_1":[[0,1,0]],
        "conc_ratio_min_2":[[1,0,0]],
        "conc_ratio_max_2":[[0,0,1]]
    }
    atoms = bulk( "Al" )
    N = 4
    atoms = atoms*(N,N,N)

    ceBulk = BulkCrystal( "fcc", 4.05, [N,N,N], 1, [["Al","Mg","Si"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
    if ( option == "generateNew" ):
        struc_generator = GenerateStructures( ceBulk, struct_per_gen=5 )
        struc_generator.generate_probe_structure( num_steps=100 )

if __name__ == "__main__":
    main( sys.argv[1:] )
