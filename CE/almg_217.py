import sys
from ase.spacegroup import crystal
from ase.visualize import view
from ase.ce.settings_bulk import BulkSpacegroup
from ase.ce.newStruct import GenerateStructures
from almg_bcc_ce import insert_specific_structure
from ase.io import read
from ase.db import connect
import numpy as np
from ase.ce import Evaluate
from atomtools.ce import ECIPlotter
import json
from matplotlib import pyplot as plt
from cemc.tools import GSFinder
from ase.io import write
from ase.units import mol, kJ
from ase.calculators.cluster_expansion import ClusterExpansion
from ase.ce import CorrFunction

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
    #cf = CorrFunction(bs)
    #cf.reconfig_db_entries()
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
    elif ( option == "allgs" ):
        find_all_gs(bs,struct_gen)

def find_all_gs(BC, struct_gen, mg_conc_min=0.5,mg_conc_max=0.7):
    mg_concs = np.linspace(mg_conc_min,mg_conc_max,20)
    insert_count = 0
    for mg_conc in mg_concs:
        fname = find_gs(BC,mg_conc)
        try:
            struct_gen.insert_structure(init_struct=fname)
            insert_count += 1
        except Exception as exc:
            print (str(exc))
    print ("Inserted {} new structures".format(insert_count))

def find_gs( BC, mg_conc ):
    composition = {
        "Mg":mg_conc,
        "Al":1.0-mg_conc
    }

    eci_fname = "data/{}.json".format(db_name.split(".")[0])
    print ("Reading ECIs from {}".format(eci_fname))
    with open( eci_fname, 'r') as infile:
        ecis = json.load( infile )

    #T = [1000,800,600,400,300,200,100,50,20,15,10,5,1]
    T = np.logspace(0,3,30)[::-1]
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
    return outfname

def evalCE( BC):
    lambs = np.logspace(-6,1,num=50)
    cvs = []
    #select_cond = None
    select_cond = [("in_conc_range","=","1")]
    for i in range(len(lambs)):
        print (lambs[i])
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1", select_cond=select_cond )

        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    print ("Selected penalization value: {}".format(lambs[indx]))
    lamb = float(lambs[indx])
    evaluator = Evaluate( BC, lamb=lamb, penalty="l1", select_cond=select_cond )

    print ( evaluator.cf_matrix[:,1] )
    eci_name = evaluator.get_cluster_name_eci_dict
    evaluator._get_e_predict()
    e_pred = evaluator.e_pred
    e_dft = evaluator.e_dft
    print (eci_name)
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot( show_names=True )

    # Formation enthalpy
    ref_eng_fcc ={"Al":-3.73667187,"Mg":-1.59090625}
    e_form_dft = []
    e_form_ce = []
    mg_concs = []
    db = connect( BC.db_name )
    calc = ClusterExpansion( BC, cluster_name_eci=eci_name )
    for i,row in enumerate(db.select(converged=1,in_conc_range=1)):
        energy = row.energy/row.natoms
        count = row.count_atoms()
        energy = e_dft[i]
        e_ce = e_pred[i]
        xmg = 0.0
        for key,value in count.iteritems():
            count[key] /= float(row.natoms)
            energy -= ref_eng_fcc[key]*count[key]
            e_ce -= ref_eng_fcc[key]*count[key]
        e_form_dft.append( energy )
        e_form_ce.append( e_ce )
        if ( "Mg" in count.keys() ):
            xmg = count["Mg"]
        mg_concs.append(xmg)

    e_form_dft = np.array(e_form_dft)
    e_form_ce = np.array( e_form_ce )
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( mg_concs, e_form_dft*mol/kJ, "o",mfc="none" )
    ax.plot( mg_concs, e_form_ce*mol/kJ, "x" )
    ax.axhline(0.0,ls="--")

    eci_fname = "data/{}.json".format(db_name.split(".")[0])
    with open( eci_fname, 'w') as outfile:
        json.dump( eci_name, outfile )
    print ("ECIs stored in %s"%(eci_fname))
    plt.show()

if __name__ == "__main__":
    main( sys.argv[1:] )
