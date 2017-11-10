import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
sys.path.insert(1,"/home/davidkl/Documents/GPAWTutorial/CE_extensions")
sys.path.append("/usr/local/lib/python2.7/dist-packages/pymatgen/cli/")
import ase
print (ase.__file__)
from ase.ce.settings import BulkCrystal
from ase.ce.evaluate import Evaluate
from ase.build import bulk
from ase.ce.newStruct import GenerateStructures
from evaluateL1min import EvaluateL1min
from ase.ce.evaluate import Evaluate
from convergence import ConvergenceCheck
import matplotlib as mpl
mpl.rcParams["svg.fonttype"]="none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import pickle
from ase.visualize import view
from plot_eci import ECIPlotter

SELECTED_ECI= "selectedEci.pkl"
def main( argv ):
    db_name = "ce_hydrostatic.db"
    option = argv[0]
    conc_args = {
        "conc_ratio_min_1":[[60,4]],
        "conc_ratio_max_1":[[64,0]],
    }
    atoms = bulk( "Al" )
    N = 4
    atoms = atoms*(N,N,N)

    ceBulk = BulkCrystal( "fcc", 4.05, [N,N,N], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
    if ( option == "generateNew" ):
        struc_generator = GenerateStructures( ceBulk, struct_per_gen=5 )
        view(atoms)
        exit()
        struc_generator.generate_probe_structure( num_steps=1000 )
    elif ( option == "evaluate" ):
        evalCE( ceBulk )

def evalCE( BC):
    evaluator = Evaluate( BC, lamb=9E-4, penalty="l1" )
    eci_name = evaluator.get_cluster_name_eci_dict
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot( tight=True )
    plt.show()

    #evaluator.plot_selected_eci()

    """
    convCheck = ConvergenceCheck( evaluator )
    print ( convCheck.converged() )
    convCheck.plot_cv_score()
    convCheck.plot_energy_with_gen_info()
    plt.show()
    """

if __name__ == "__main__":
    main( sys.argv[1:] )
