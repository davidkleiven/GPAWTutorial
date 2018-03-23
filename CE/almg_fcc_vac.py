from ase.ce import BulkCrystal, CorrFunction, Evaluate
from atomtools.ce import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
import json

db_name = "almg_fcc_vac.db"
eci_fname = "data/almg_fcc_vac_eci.json"
def main():
    conc_args = {
          "conc_ratio_min_1":[[1,0,0]],
          "conc_ratio_max_1":[[0,1,1]],
      }

    ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[4,4,4], basis_elements=[["Al","Mg","X"]], \
    conc_args=conc_args, db_name=db_name, max_cluster_size=4 )
    #ceBulk._get_cluster_information()
    #print (ceBulk.basis_functions)
    #cf = CorrFunction( ceBulk )
    #cf.reconfig_db_entries()
    evaluate( ceBulk )

def evaluate(BC):
    lambs = np.logspace(-7,-1,num=50)
    cvs = []
    for i in range(len(lambs)):
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1" )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    print ("Selected penalization: {}".format(lambs[indx]))
    evaluator = Evaluate( BC, lamb=float(lambs[indx]), penalty="l1" )
    eci_name = evaluator.get_cluster_name_eci_dict
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot()
    plt.show()

    with open(eci_fname,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_fname))

if __name__ == "__main__":
    main()
