import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
sys.path.append("/usr/local/lib/python2.7/dist-packages/pymatgen/cli/")
import ase
print (ase.__file__)
from ase.ce.settings import BulkCrystal
from ase.build import bulk
from ase.ce.newStruct import GenerateStructures

def main():
    conc_args = {
        "conc_ratio_min_1":[[0,1]],
        "conc_ratio_max_1":[[1,0]],
    }
    atoms = bulk( "Al" )
    N = 4
    atoms = atoms*(N,N,N)

    db_name = "ceTest.db"
    ceBulk = BulkCrystal( "fcc", 4.05, [N,N,N], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4 )
    #print (ceBulk.conc_matrix)
    struc_generator = GenerateStructures( ceBulk, struct_per_gen=3 )

    struc_generator.generate()

if __name__ == "__main__":
    main()
