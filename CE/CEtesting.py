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
#from plot_corr_matrix import CovariancePlot
from atomtools.ce.corrmatrix import CovariancePlot
from convex_hull_plotter import QHull
from ase.ce.corrFunc import CorrFunction
import json
from atomtools.ce.evaluate_deviation import EvaluateDeviation
from atomtools.ce.phonon_ce_eval import PhononEval
from atomtools.ce.population_variance import PopulationVariance

SELECTED_ECI= "selectedEci.pkl"
db_name = "ce_hydrostatic.db"
#db_name = "ce_hydrostatic_phonons.db"
def main( argv ):
    option = argv[0]
    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]],
    }
    atoms = bulk( "Al" )
    N = 4
    atoms = atoms*(N,N,N)

    ceBulk = BulkCrystal( "fcc", 4.05, None, [N,N,N], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
    struc_generator = GenerateStructures( ceBulk, struct_per_gen=5 )
    if ( option == "generateNew" ):
        struc_generator.generate_probe_structure()
    elif ( option == "evaluate" ):
        evalCE( ceBulk )
        #eval_phonons( ceBulk )
    elif ( option == "popstat" ):
        find_pop_statistics( ceBulk )
    elif ( option == "insert" ):
        atoms = bulk("Mg","fcc",a=4.05)
        atoms = atoms*(N,N,N)
        al_indices = np.random.randint(low=0,high=64,size=14)
        for indx in al_indices:
            atoms[indx].symbol = "Al"
        view(atoms)
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

def evalCE( BC):
    lambs = np.logspace(-7,-1,num=50)
    print (lambs)
    cvs = []
    for i in range(len(lambs)):
        print (lambs[i])
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1" )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    print ("Selected penalization value: {}".format(lambs[indx]))
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
    plotter.plot( show_names=True )

    cov_plotter = CovariancePlot(evaluator, constant_term_column=0)
    cov_plotter.plot()
    cov_plotter.plot_corr_func_coverage()
    eci_fname = "data/almg_eci.json"
    with open( eci_fname, 'w') as outfile:
        json.dump( eci_name, outfile )
    print ("ECIs stored in %s"%(eci_fname))

    #qhull = QHull( db_name )
    #qhull.plot( "Al" )
    plt.show()

def eval_phonons( ceBulk ):
    lambs = np.logspace(-2,3,num=50)
    temps = [800,700,600,500,400,300,200,100]
    for T in temps:
        cvs = []
        for i in range(len(lambs)):
            print ("%d of %d"%(i,len(lambs)) )
            pce = PhononEval( ceBulk, lamb=lambs[i], penalty="l1" )
            pce.T = T
            cvs.append(pce._cv_loo() )
        indx = np.argmin(cvs)
        l = lambs[indx]
        pce = PhononEval( ceBulk, lamb=l, penalty="l1" )
        pce.T = T
        eci_name = pce.get_cluster_name_eci_dict
        pce.plot_energy()
        plotter = ECIPlotter(eci_name)
        plotter.plot( show_names=True )
        eci_fname = "data/almg_eci_Fvib%d.json"%(T)
        with open( eci_fname, 'w') as outfile:
            json.dump( eci_name, outfile )
        print ( "Phonon ECIs written to %s"%(eci_fname) )
    plt.show()

def find_pop_statistics( ceBulk ):
    popvar = PopulationVariance( ceBulk )
    cov, mean = popvar.estimate(  n_probe_structures=10000, fname="data/almg_covmean.json" )

if __name__ == "__main__":
    main( sys.argv[1:] )
