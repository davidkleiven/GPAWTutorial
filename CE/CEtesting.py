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
from ceext import evaluate_prior as ep
from ceext import penalization as pen
import numpy as np
from plot_corr_matrix import CovariancePlot
from convex_hull_plotter import QHull

SELECTED_ECI= "selectedEci.pkl"
db_name = "ce_hydrostatic.db"
def main( argv ):
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
        struc_generator.generate_probe_structure( num_steps=100 )
    elif ( option == "evaluate" ):
        evalCE( ceBulk )

def evalCE( BC):
    lambs = np.logspace(-7,-1,num=50)
    print (lambs)
    cvs = []
    for i in range(len(lambs)):
        print (lambs[i])
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1" )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    evaluator = Evaluate( BC, lamb=float(lambs[indx]), penalty="l1" )
    #evaluator = ep.EvaluatePrior(BC, selection={"nclusters":5} )
    #cnames = evaluator.cluster_names
    #l1 = pen.L1(9E-4)
    #evaluator.add_penalization( [pen.volume_penalization(1E-14,cnames,penalization="l2"),pen.number_atoms_penalization(1E-14,cnames,penalization="l2")] )
    #evaluator.add_penalization( [pen.number_atoms_penalization(0.0001,cnames)] )
    #evaluator.estimate_hyper_params()
    eci_name = evaluator.get_cluster_name_eci_dict
    print (eci_name)
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot( show_names=False )

    cov_plotter = CovariancePlot(evaluator, constant_term_column=0)
    cov_plotter.plot()

    #qhull = QHull( db_name )
    #qhull.plot( "Al" )
    plt.show()


if __name__ == "__main__":
    main( sys.argv[1:] )
