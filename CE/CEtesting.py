import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
sys.path.append("/usr/local/lib/python2.7/dist-packages/pymatgen/cli/")
import ase
print (ase.__file__)
from ase.ce.settings import BulkCrystal
from ase.build import bulk
from ase.ce.newStruct import GenerateStructures
from ase.ce.evaluateL1min import EvaluateL1min
from ase.ce.evaluate import Evaluate
from matplotlib import pyplot as plt

def main( argv ):
    db_name = "ceTest.db"
    option = argv[0]
    if ( option == "generateNew" ):
        conc_args = {
            "conc_ratio_min_1":[[52,12]],
            "conc_ratio_max_1":[[64,0]],
        }
        atoms = bulk( "Al" )
        N = 4
        atoms = atoms*(N,N,N)

        ceBulk = BulkCrystal( "fcc", 4.05, [N,N,N], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4 )
        print (ceBulk.cluster_names)

        struc_generator = GenerateStructures( ceBulk, struct_per_gen=6 )

        struc_generator.generate()
    elif ( option == "evaluate" ):
        evalCE( db_name )



def evalCE( db_name ):
    evaluator = EvaluateL1min( db_name, threshold=0.001, alpha=0.001, use_scipy_linprog=False )
    eci = evaluator.get_eci()
    print (evaluator.selected_cluster_names)
    evaluator.plot_energy()
    evaluator.plot_selected_eci()
    evaluator.plot_energy()

if __name__ == "__main__":
    main( sys.argv[1:] )
