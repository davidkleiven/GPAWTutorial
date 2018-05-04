import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
sys.path.insert(2,"/home/dkleiven/Documents/aseJin")
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
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.db import connect
from ase.units import mol, kJ
from atomtools.ce import CVScoreHistory

eci_fname = "data/almgsi_fcc_eci.json"
db_name = "almgsi.db"
def main(argv):
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
    orig_spin_dict = {
        "Mg":1.0,
        "Si":-1.0,
        "Al":0.0
    }

    ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[N,N,N], basis_elements=[["Mg","Si","Al",]], \
    conc_args=conc_args, db_name=db_name, max_cluster_size=4 )
    #ceBulk.spin_dict = orig_spin_dict
    #ceBulk.basis_functions = ceBulk._get_basis_functions()
    #ceBulk._get_cluster_information()
    print (ceBulk.basis_functions)
    cf = CorrFunction( ceBulk )
    #cf.reconfig_db_entries( select_cond=[("id",">=","2315")])
    #exit()
    struc_generator = GenerateStructures( ceBulk, struct_per_gen=10 )
    if ( option == "generateNew" ):
        struc_generator.generate_probe_structure()
    elif ( option == "eval" ):
        evaluate(ceBulk)
    elif ( option == "insert" ):
        fname = argv[1]
        atoms = read(fname)
        struc_generator.insert_structure( init_struct=fname )
        #insert_specific_structure( ceBulk, struc_generator, atoms )
    elif ( option == "formation" ):
        enthalpy_of_formation(ceBulk)
    elif ( option == "gsstruct" ):
        if ( len(argv) != 3 ):
            raise ValueError( "If option is gsstruct. The arguments has to be gsstruct mg_conc si_conc" )
        mg_conc = float( argv[1] )
        si_conc = float( argv[2] )
        find_gs_structure( ceBulk, mg_conc, si_conc )
    elif ( option == "update_conc_range" ):
        update_in_conc_range()
    elif ( option == "allgs" ):
        find_all_gs( ceBulk, struc_generator )
    elif( option == "cv_hist" ):
        lambdas = np.logspace(-7,-3,8)
        history = CVScoreHistory(setting=ceBulk, penalization="L1", select_cond=[("in_conc_range","=","1")] )
        history.get_history( lambdas=lambdas )
        history.plot()
        plt.show()


def update_in_conc_range():
    db = connect( db_name )
    for row in db.select():
        at_count = row.count_atoms()
        conc_si = 0.0
        if ( "Si" in at_count.keys() ):
            conc_si = at_count["Si"]/float(row.natoms)
        if ( conc_si > 0.32 ):
            db.update( row.id, in_conc_range=0 )

def evaluate(BC):
    lambs = np.logspace(-5,-4,num=50)
    cvs = []
    s_cond = [("in_conc_range","=","1")]
    for i in range(len(lambs)):
        print (lambs[i])
        evaluator = Evaluate( BC, lamb=float(lambs[i]), penalty="l1", select_cond=s_cond )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    print ("Selected penalization: {}".format(lambs[indx]))
    evaluator = Evaluate( BC, lamb=float(lambs[indx]), penalty="l1", select_cond=s_cond )
    eci_name = evaluator.get_cluster_name_eci_dict
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot()
    plt.show()

    with open(eci_fname,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_fname))

def enthalpy_of_formation(ceBulk):
    with open( "data/almgsi_fcc_eci.json", 'r' ) as infile:
        ecis = json.load(infile)

    db = connect( ceBulk.db_name )
    ce_ase_calc = ClusterExpansion( ceBulk, cluster_name_eci=ecis )
    #atoms = bulk("Al")*(4,4,4)
    #atoms.set_calculator(ce_ase_calc)
    ceBulk.atoms.set_calculator(ce_ase_calc)

    #ref_eng_al = db.get(formula="Al64",converged=1).energy
    ref_eng_al = -3.73712125264
    ref_eng_mg = db.get(formula="Mg1").energy
    ref_eng_si = db.get(formula="Si1").energy
    enthalpy_dft = []
    enthalpy_ce = []
    conc_mg = []
    conc_si = []
    formulas = []
    for row in db.select(converged=1):
        e_dft = row.energy/row.natoms
        count = row.count_atoms()
        c_mg = 0.0
        c_si = 0.0
        if ( "Mg" in count.keys() ):
            c_mg = float(count["Mg"])/row.natoms
        if ( "Si" in count.keys() ):
            c_si = float(count["Si"])/row.natoms
        c_al = 1.0-c_mg-c_si
        atoms = row.toatoms()
        #atoms.set_calculator(ce_ase_calc)
        E_ce = atoms.get_potential_energy()/row.natoms
        for i in range(len(ceBulk.atoms)):
            if ( len(atoms) == 1 ):
                ceBulk.atoms[i].symbol = atoms[0].symbol
            else:
                ceBulk.atoms[i].symbol = atoms[i].symbol
        H_dft = e_dft - c_al*ref_eng_al - c_mg*ref_eng_mg - c_si*ref_eng_si
        H_ce = E_ce - c_al*ref_eng_al - c_mg*ref_eng_mg - c_si*ref_eng_si
        enthalpy_dft.append( H_dft )
        enthalpy_ce.append( H_ce )
        conc_mg.append( c_mg )
        conc_si.append( c_si )
        formulas.append( row.formula )
    conc_mg = np.array( conc_mg )
    conc_si = np.array( conc_si )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    x = conc_mg/(conc_mg+conc_si)
    enthalpy_dft = np.array(enthalpy_dft)*mol/kJ
    enthalpy_ce = np.array(enthalpy_ce)*mol/kJ
    ax.plot( x, enthalpy_dft, "o", mfc="none" )
    ax.plot( x, enthalpy_ce, "x" )
    ax.set_xlabel( "\$c_{Mg}/(c_{Mg}+c_{Si})")
    ax.set_ylabel( "Enthalpy of formation (kJ/mol)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    for i in range(len(formulas)):
        ax.text( x[i], enthalpy_dft[i], formulas[i] )
    plt.show()

def find_all_gs(ceBulk,struct_gen):
    fname = "almgsi_gs_search.csv"
    mg_concs, si_concs = np.loadtxt( fname, delimiter=",", unpack=True )
    n_inserted = 0
    for xmg,xsi in zip(mg_concs,si_concs):
        fname = find_gs_structure( ceBulk, xmg,xsi )
        try:
            struct_gen.insert_structure( init_struct=fname )
            n_inserted += 1
        except Exception as exc:
            print (str(exc))

    print ("Inserted {} new ground state structures".format(n_inserted))

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

    temperatures = [800,700,600,500,400,300,200,175,150,125,100,75,50,25,20,10]
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
    gs_fname = fname
    write( fname, lowest_struct.lowest_energy_atoms )
    print ("Lowest energy found: {}".format( lowest_struct.lowest_energy))
    print ("GS structure saved to %s"%(fname) )
    fname = "data/cf_functions_gs%s.csv"%(formula)
    cf = calc.get_cf()
    with open(fname,'w') as outfile:
        for key,value in cf.iteritems():
            outfile.write( "{},{}\n".format(key,value))
    print ("CFs saved to %s"%(fname))
    return gs_fname

if __name__ == "__main__":
    main( sys.argv[1:] )
