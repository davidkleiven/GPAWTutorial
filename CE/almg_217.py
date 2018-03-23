import sys
from ase.spacegroup import crystal
from ase.visualize import view
from ase.ce.settings_bulk import BulkSpacegroup
from ase.ce.newStruct import GenerateStructures
from almg_bcc_ce import insert_specific_structure
from ase.io import read
import numpy as np
from ase.ce import Evaluate
from atomtools.ce import ECIPlotter
import json
from matplotlib import pyplot as plt
from cemc.tools import GSFinder
from ase.io import write

def get_atoms():
    # https://materials.springer.com/isp/crystallographic/docs/sd_0453869
    a = 10.553
    b = 10.553
    c = 10.553
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a,b,c,alpha,beta,gamma]
    symbols = ["Mg","Mg","Mg","Al"]
    #symbols = ["Al","Al","Al","Al"]
    basis = [(0,0,0),(0.324,0.324,0.324),(0.3582,0.3582,0.0393),(0.0954,0.0954,0.2725)]
    atoms = crystal( symbols, spacegroup=217, cellpar=cellpar, basis=basis)
    return atoms, cellpar,symbols,basis

def get_atoms_pure():
    # https://materials.springer.com/isp/crystallographic/docs/sd_0453869
    a = 9.993
    b = 9.993
    c = 9.993
    alpha = 90
    beta = 90
    gamma = 90
    cellpar = [a,b,c,alpha,beta,gamma]
    symbols = ["Mg","Mg","Mg","Al"]
    #symbols = ["Al","Al","Al","Al"]
    basis = [(0,0,0),(0.332,0.332,0.332),(0.362,0.362,0.051),(0.0949,0.0949,0.2864)]
    atoms = crystal( symbols, spacegroup=217, cellpar=cellpar, basis=basis)
    return atoms, cellpar,symbols,basis

db_name = "almg_217.db"
def main( argv ):
    option = argv[0]
    atoms, cellpar, symbols, basis = get_atoms()
    conc_args = {
        "conc_ratio_min_1":[[1,0],[1,0],[1,0],[1,0]],
        "conc_ratio_max_1":[[0,1],[0,1],[0,1],[0,1]]
    }
    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]]
    }
    basis_elements = [["Al","Mg"],["Al","Mg"],["Al","Mg"],["Al","Mg"]]
    bs = BulkSpacegroup( basis_elements=basis_elements, basis=basis, spacegroup=217, cellpar=cellpar, conc_args=conc_args,
    max_cluster_size=4, db_name=db_name, size=[1,1,1], grouped_basis=[[0,1,2,3]] )
    #bs.reconfigure_settings()
    print (bs.basis_functions)

    struct_gen = GenerateStructures( bs, struct_per_gen=10 )

    if ( option == "newstruct" ):
        struct_gen.generate_probe_structure( num_steps=100 )
    elif ( option == "insert" ):
        fname = argv[1]
        atoms = read( fname )
        struct_gen.insert_structure(init_struct=fname)
        #insert_specific_structure( bs, struct_gen, atoms )
    elif ( option == "eval" ):
        evalCE(bs)
    elif ( option == "gs" ):
        mg_conc = float( argv[1] )
        find_gs( bs, mg_conc )

def find_gs( BC, mg_conc ):
    composition = {
        "Mg":mg_conc,
        "Al":1.0-mg_conc
    }

    eci_fname = "data/{}.json".format(db_name.split(".")[0])
    print ("Reading ECIs from {}".format(eci_fname))
    with open( eci_fname, 'r') as infile:
        ecis = json.load( infile )

    T = [1000,800,600,400,200,100,50,20,10,5,1]
    n_steps_per = 5000
    gsfinder = GSFinder()
    result = gsfinder.get_gs( BC, ecis, composition=composition, temps=T, n_steps_per_temp=n_steps_per )
    atoms = result["atoms"]
    formula = atoms.get_chemical_formula()
    outfname = "data/gs_217_{}_{}mev.xyz".format( formula, int(result["energy"]*1000) )
    write( outfname, atoms )
    print ("GS energy: {} eV".format(result["energy"]) )
    print ("Structure written to {}".format(outfname) )
    print (result["cf"])

def evalCE( BC):
    lambs = np.logspace(-7,-1,num=50)
    cvs = []
    for i in range(len(lambs)):
        print (lambs[i])
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1" )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    print ("Selected penalization value: {}".format(lambs[indx]))
    evaluator = Evaluate( BC, lamb=float(lambs[indx]), penalty="l1" )
    print ( evaluator.cf_matrix[:,1] )
    eci_name = evaluator.get_cluster_name_eci_dict
    print (eci_name)
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot( show_names=True )

    eci_fname = "data/{}.json".format(db_name.split(".")[0])
    with open( eci_fname, 'w') as outfile:
        json.dump( eci_name, outfile )
    print ("ECIs stored in %s"%(eci_fname))
    plt.show()

if __name__ == "__main__":
    main( sys.argv[1:] )
