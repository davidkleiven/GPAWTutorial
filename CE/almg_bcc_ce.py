import sys
from ase.ce.settings import BulkCrystal
from ase.ce.evaluate import Evaluate
from ase.ce.newStruct import GenerateStructures
from ase.io import read
from ase.ce.corrFunc import CorrFunction
from atomtools.ce.eciplotter import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
from cemc.wanglandau.ce_calculator import CE
from cemc.mcmc import montecarlo as mc
from cemc.mcmc import mc_observers as mcobs
import json

def main( argv ):
    db_name = "almg_bcc.db"
    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]]
    }
    ceBulk = BulkCrystal( "bcc", 3.3, None, [4,4,4], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=4, reconf_db=False)
    struc_generator = GenerateStructures( ceBulk, struct_per_gen=5 )

    if ( len(argv) == 0 ):
        return

    option = argv[0]

    if ( option == "insert" ):
        fname = argv[1]
        atoms = read(fname)
        insert_specific_structure( ceBulk, struc_generator, atoms )
    elif ( option == "eval" ):
        evaluate( ceBulk )
    elif ( option == "gsstruct" ):
        find_gs_structures( ceBulk, struc_generator, at_step=4 )

def insert_specific_structure( ceBulk, struct_gen, atoms ):
    cf = CorrFunction( ceBulk )
    kvp = cf.get_cf(atoms)
    conc = struct_gen.find_concentration(atoms)
    print (conc)
    if ( struct_gen.exists_in_db(atoms,conc[0],conc[1]) ):
        raise RuntimeError( "The passed structure already exists in the database" )
    kvp = struct_gen.get_kvp( atoms, kvp, conc[0], conc[1] )
    struct_gen.db.write(atoms,kvp)

def find_gs_structures( ceBulk, struct_gen, at_step=4 ):
    with open( "data/almg_eci_bcc.json", 'r' ) as infile:
        ecis = json.load(infile)
    calc = CE( ceBulk, ecis )
    ceBulk.atoms.set_calculator( calc )

    mg_atoms = [at_step*i for i in range(1,int(64/at_step)-at_step)]
    temps = [800,600,400,200,100,50,25,20,19,18,16,17,15,14,13,12,11,10,9,8,7,6]
    n_steps_per = 6000
    num_inserted = 0
    for n_mg in mg_atoms:
        c_mg = n_mg/64.0
        composition = {"Mg":c_mg, "Al":1.0-c_mg}
        ceBulk.atoms._calc.set_composition( composition )
        lowest_struct = mcobs.LowestEnergyStructure( calc, None )
        print (calc.get_cf())
        for T in temps:
            print ("Temperature {}. Formula: {}".format(T, ceBulk.atoms.get_chemical_formula()) )
            mc_obj = mc.Montecarlo( ceBulk.atoms, T )
            lowest_struct.mc_obj = mc_obj
            mc_obj.attach( lowest_struct )
            mc_obj.runMC( steps=n_steps_per, verbose=False )
            thermo = mc_obj.get_thermodynamic()
            print ("Average energy: {}".format(thermo["energy"]) )
        try:
            insert_specific_structure( ceBulk, struct_gen, lowest_struct.atoms )
            num_inserted += 1
        except:
            pass
    print ( "Inserted {} new structures".format(num_inserted) )



def evaluate( BC ):
    lambs = np.logspace( -7, -1, 50 )
    cvs = []
    for i in range(len(lambs)):
        print (lambs[i])
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1" )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    print ("Minimum CV at lambda: {}".format(lambs[indx]))
    evaluator = Evaluate( BC, lamb=float(lambs[indx]), penalty="l1" )
    eci_name = evaluator.get_cluster_name_eci_dict
    print (eci_name)
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot( show_names=True )
    plt.show()

    fname = "data/almg_eci_bcc.json"
    with open( fname, 'w' ) as outfile:
        json.dump( eci_name, outfile )
    print ( "ECIs written to {}".format(fname) )

if __name__ == "__main__":
    main( sys.argv[1:] )
