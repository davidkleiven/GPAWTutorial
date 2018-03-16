import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
sys.path.insert(1,"/home/dkleiven/Documents/aseJin")
sys.path.append("/home/davidkl/Documents/GPAWTutorial/CE_extensions")
sys.path.append("/home/dkleiven/Documents/GPAWTutorials/CE_extensions")
from ase.build import bulk
from ase.ce.settings_bulk import BulkCrystal
from ase.ce.corrFunc import CorrFunction
from ase.ce.newStruct import GenerateStructures
from atomtools.ce.corrmatrix import CovariancePlot
#from convex_hull_plotter import QHull
from ase.ce.evaluate import Evaluate
from plot_eci import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
from cemc.wanglandau.ce_calculator import CE
from cemc.mcmc import mc_observers as mcobs
import json
from cemc.mcmc import montecarlo as mc
from ase.io import write, read
from almg_bcc_ce import insert_specific_structure

eci_fname = "data/almgsi_fcc_eci.json"
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

    ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[N,N,N], basis_elements=[["Al","Mg","Si"]], \
    conc_args=conc_args, db_name=db_name, max_cluster_size=4 )
    ceBulk._get_cluster_information()
    print (ceBulk.basis_functions)
    cf = CorrFunction( ceBulk )
    #cf.reconfig_db_entries()
    #exit()
    struc_generator = GenerateStructures( ceBulk, struct_per_gen=5 )
    if ( option == "generateNew" ):
        struc_generator.generate_probe_structure()
    elif ( option == "eval" ):
        evaluate(ceBulk)
    elif ( option == "insert" ):
        fname = argv[1]
        atoms = read(fname)
        insert_specific_structure( ceBulk, struc_generator, atoms )
    elif ( option == "gsstruct" ):
        if ( len(argv) != 3 ):
            raise ValueError( "If option is gsstruct. The arguments has to be gsstruct mg_conc si_conc" )
        mg_conc = float( argv[1] )
        si_conc = float( argv[2] )
        find_gs_structure( ceBulk, mg_conc, si_conc )

def evaluate(BC):
    lambs = np.logspace(-7,-1,num=50)
    cvs = []
    for i in range(len(lambs)):
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1" )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    evaluator = Evaluate( BC, lamb=float(lambs[indx]), penalty="l1" )
    eci_name = evaluator.get_cluster_name_eci_dict
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot()
    plt.show()

    with open(eci_fname,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_fname))

def find_gs_structure( ceBulk, mg_conc, si_conc ):
    """
    Finds a GS structure
    """
    with open(eci_fname,'r') as infile:
        eci = json.load( infile )

    conc = {
        "Mg":mg_conc,
        "Si":si_conc,
        "Al":1.0-mg_conc-si_conc
    }

    calc = CE( ceBulk, eci )
    ceBulk.atoms.set_calculator(calc)
    calc.set_composition( conc )

    temperatures = [800,700,600,500,400,300,200,100,50,20,10]
    nsteps = 10000
    lowest_struct = mcobs.LowestEnergyStructure( calc, None )
    formula = ceBulk.atoms.get_chemical_formula()
    for T in temperatures:
        print ("Temperature {}".format(T) )
        mc_obj = mc.Montecarlo( ceBulk.atoms, T )
        lowest_struct.mc_obj = mc_obj
        mc_obj.attach( lowest_struct )
        mc_obj.runMC( steps=nsteps, verbose=False )
        thermo = mc_obj.get_thermodynamic()
        print ("Mean energy: {}".format(thermo["energy"]))
    fname = "data/gs_structure%s.xyz"%(formula)
    write( fname, lowest_struct.lowest_energy_atoms )
    print ("Lowest energy found: {}".format( lowest_struct.lowest_energy))
    print ("GS structure saved to %s"%(fname) )
    fname = "data/cf_functions_gs%s.csv"%(formula)
    cf = calc.get_cf()
    with open(fname,'w') as outfile:
        for key,value in cf.iteritems():
            outfile.write( "{},{}\n".format(key,value))
    print ("CFs saved to %s"%(fname))

if __name__ == "__main__":
    main( sys.argv[1:] )
