import sys
from ase.ce.settings import BulkCrystal
from ase.ce.newStruct import GenerateStructures
from ase.io import read
from ase.ce.corrFunc import CorrFunction

def main( argv ):
    db_name = "almg_bcc.db"
    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]]
    }
    ceBulk = BulkCrystal( "bcc", 3.3, None, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
    struc_generator = GenerateStructures( ceBulk, struct_per_gen=5 )

    if ( len(argv) == 0 ):
        return

    option = argv[0]

    if ( option == "insert" ):
        fname = argv[1]
        atoms = read(fname)
        insert_specific_structure( ceBulk, struc_generator, atoms )

def insert_specific_structure( ceBulk, struct_gen, atoms ):
    cf = CorrFunction( ceBulk )
    kvp = cf.get_cf(atoms)
    conc = struct_gen.find_concentration(atoms)
    print (conc)
    if ( struct_gen.exists_in_db(atoms,conc[0],conc[1]) ):
        raise RuntimeError( "The passed structure already exists in the database" )
    kvp = struct_gen.get_kvp( atoms, kvp, conc[0], conc[1] )
    struct_gen.db.write(atoms,kvp)

if __name__ == "__main__":
    main( sys.argv[1:] )
