import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
sys.path.append("/home/davidkl/Documents/GPAWTutorial/CE_extensions")
from ase.build import bulk
from ase.ce.settings import BulkCrystal
from ase.ce.newStruct import GenerateStructures
from plot_corr_matrix import CovariancePlot
from convex_hull_plotter import QHull
from ase.ce.evaluate import Evaluate
from plot_eci import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt

def main(argv):
    db_name = "almgsi.db"
    option = argv[0]
    conc_args = {
        "conc_ratio_min_1":[[64,0,0]],
        "conc_ratio_max_1":[[24,40,0]],
        "conc_ratio_min_2":[[64,0,0]],
        "conc_ratio_max_2":[[22,21,21]]
    }
    atoms = bulk( "Al" )
    N = 4
    atoms = atoms*(N,N,N)

    ceBulk = BulkCrystal( "fcc", 4.05, [N,N,N], 1, [["Al","Mg","Si"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
    if ( option == "generateNew" ):
        struc_generator = GenerateStructures( ceBulk, struct_per_gen=5 )
        struc_generator.generate_probe_structure()
    elif ( option == "eval" ):
        evaluate(ceBulk)

def evaluate(BC):
    lambs = np.logspace(-7,-1,num=50)
    cvs = []
    for i in range(len(lambs)):
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1" )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    evaluator = Evaluate( BC, lamb=float(lambs[indx]), penalty="l1" )
    eci_name = evaluator.get_cluster_name_eci_dict
    print (eci_name)
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot()

    cov_plotter = CovariancePlot(evaluator, constant_term_column=0)
    cov_plotter.plot()

    #qhull = QHull( db_name )
    #qhull.plot( "Al" )
    plt.show()
if __name__ == "__main__":
    main( sys.argv[1:] )
