import sys
from ase.ce import BulkCrystal, CorrFunction, Evaluate, GenerateStructures
from atomtools.ce import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
import json
from cemc.tools import GSFinder
from ase.io import read,write

db_name = "almg_fcc_vac.db"
eci_fname = "data/almg_fcc_vac_eci.json"
def main( argv ):
    option = argv[0]
    conc_args = {
          "conc_ratio_min_1":[[64,0,0]],
          "conc_ratio_max_1":[[0,48,16]],
      }

    ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[4,4,4], basis_elements=[["Al","Mg","X"]], \
    conc_args=conc_args, db_name=db_name, max_cluster_size=4 )
    #ceBulk._get_cluster_information()
    #print (ceBulk.basis_functions)
    #cf = CorrFunction( ceBulk )
    #cf.reconfig_db_entries()
    struct_gen = GenerateStructures( ceBulk, struct_per_gen=10 )
    if ( option == "eval" ):
        evaluate( ceBulk )
    elif( option == "gs" ):
        mg_conc = float(argv[1])
        vac_conc = float(argv[2])
        find_gs( ceBulk, mg_conc, vac_conc )
    elif( option == "insert" ):
        fname = argv[1]
        struct_gen.insert_structure( init_struct=fname )

def find_gs( BC, mg_conc, vac_conc ):
    composition = {
        "Mg":mg_conc,
        "X":vac_conc,
        "Al":1.0-mg_conc-vac_conc
    }

    print ("Reading ECIs from {}".format(eci_fname))
    with open( eci_fname, 'r') as infile:
        ecis = json.load( infile )

    T = [1000,800,600,400,200,100,50,20,10,5,1]
    n_steps_per = 5000
    gsfinder = GSFinder()
    result = gsfinder.get_gs( BC, ecis, composition=composition, temps=T, n_steps_per_temp=n_steps_per )
    atoms = result["atoms"]
    formula = atoms.get_chemical_formula()
    outfname = "data/gs_fcc_vac_{}_{}mev.xyz".format( formula, int(result["energy"]*1000) )
    write( outfname, atoms )
    print ("GS energy: {} eV".format(result["energy"]) )
    print ("Structure written to {}".format(outfname) )
    print (result["cf"])

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
    main( sys.argv[1:] )
