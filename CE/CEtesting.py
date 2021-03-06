import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
#sys.path.insert(1,"/home/davidkl/Documents/aseJin")
#sys.path.insert(1,"/home/davidkl/Documents/GPAWTutorial/CE_extensions")
sys.path.append("/usr/local/lib/python2.7/dist-packages/pymatgen/cli/")
import ase
print (ase.__file__)
from ase.clease import CEBulk as BulkCrystal
from ase.clease import Evaluate
from ase.build import bulk
from ase.clease import NewStructures as GenerateStructures
from ase.clease import CorrFunction
#from evaluateL1min import EvaluateL1min
#from convergence import ConvergenceCheck
import matplotlib as mpl
mpl.rcParams["svg.fonttype"]="none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import pickle
from ase.visualize import view
#from plot_eci import ECIPlotter
from atomtools.ce.eciplotter import ECIPlotter
#from ceext import evaluate_prior as ep
#from ceext import penalization as pen
import numpy as np
#from plot_corr_matrix import CovariancePlot
from atomtools.ce.corrmatrix import CovariancePlot
#from convex_hull_plotter import QHull
import json
from atomtools.ce.evaluate_deviation import EvaluateDeviation
from atomtools.ce.phonon_ce_eval import PhononEval
from atomtools.ce.population_variance import PopulationVariance
from cemc import CE
from cemc.mcmc import montecarlo as mc
from cemc.mcmc import mc_observers as mcobs
from ase.io import write, read
from ase.db import connect
from ase.calculators.clease import Clease as ClusterExpansion
import pickle as pck
from atomtools.ce import CVScoreHistory

SELECTED_ECI= "selectedEci.pkl"
#db_name = "ce_hydrostatic_phonons.db"
#db_name = "ce_hydrostatic.db"
#db_name = "ce_hydrostatic_eam_relax_effect_ideal.db"
#db_name = "almg_eam.db"

class ExcludeHighMg(object):
    def __init__( self, max_mg ):
        self.max_mg = max_mg
    def __call__(self, count ):
        if ( not "Mg" in count.keys() ):
            return True
        if ( count["Mg"] > self.max_mg ):
            return False
        return True

def main( argv ):
    lattice = argv[0]
    global db_name
    conc_args = {
        "conc_ratio_min_1":[[1,0]],
        "conc_ratio_max_1":[[0,1]],
    }
    if ( lattice == "fcc" ):
        db_name = "ce_hydrostatic_new_config.db"
        db_name = "ce_hydrostatic_only_relaxed.db"
        pickle_file = "data/BC_fcc.pkl"
        ceBulk = BulkCrystal( crystalstructure="fcc", a=4.05, size=[4,4,4], \
                              basis_elements=[["Al","Mg"]], conc_args=conc_args,
                              db_name=db_name,
                              max_cluster_size=4)
    elif ( lattice == "bcc" ):
        db_name = "almg_bcc.db"
        pickle_file = "data/BC_bcc.pkl"
        ceBulk = BulkCrystal( crystalstructure="bcc", a=3.3, size=[4,4,4], \
                              basis_elements=[["Al","Mg"]], conc_args=conc_args,
                              db_name=db_name,
                              max_cluster_size=4)
    option = argv[1]
    atoms = bulk( "Al" )
    N = 4
    atoms = atoms*(N,N,N)


    #ceBulk.basis_functions[0]["Al"] = 1.0
    #ceBulk.basis_functions[0]["Mg"] = -1.0
    #ceBulk._get_cluster_information()
    print (ceBulk.basis_functions)
    #cf = CorrFunction(ceBulk)
    #cf.reconfig_db_entries(select_cond=[("calculator","=","unknown")])
    #exit()
    #ceBulk.view_clusters()
    #ceBulk._get_cluster_information()
    #cf_names = cf.get_cf( ceBulk.atoms )
    #cf_names = cf.get_cf_by_cluster_names( ceBulk.atoms, ["c2_1000_1_00"])
    #print (cf_names)
    #print (ceBulk.cluster_names)
    #cf.reconfig_db_entries()
    #exit()
    with open(pickle_file,'wb') as outfile:
        pck.dump( ceBulk, outfile )
    #struc_generator = GenerateStructures( ceBulk, struct_per_gen=5 )
    struc_generator = GenerateStructures( ceBulk )
    if ( option == "generateNew" ):
        struc_generator.generate_probe_structure()
    elif ( option == "evaluate" ):
        evalCE( ceBulk )
    elif ( option == "phonons" ):
        eval_phonons( ceBulk )
    elif ( option == "popstat" ):
        find_pop_statistics( ceBulk )
    elif ( option == "gsstruct" ):
        find_gs_structure( ceBulk, float(argv[2]) )
    elif ( option == "formation" ):
        enthalpy_of_formation( ceBulk )
    elif ( option == "insert" ):
        if ( len(argv) == 3 ):
            fname = argv[2]
        else:
            raise ValueError( "No xyz filename given!" )
        atoms = read(fname)
        insert_specific_structure( ceBulk, struc_generator, atoms )
    elif( option == "cv_hist" ):
        lambdas = np.logspace(-7,-3,8)
        history = CVScoreHistory(setting=ceBulk, penalization="L1" )
        history.get_history( lambdas=lambdas )
        history.plot()
        plt.show()

def insert_specific_structure( ceBulk, struct_gen, atoms ):
    cf = CorrFunction( ceBulk )
    kvp = cf.get_cf(atoms)
    conc = struct_gen.find_concentration(atoms)
    print (conc)
    if ( struct_gen.exists_in_db(atoms,conc[0],conc[1]) ):
        raise RuntimeError( "The passed structure already exists in the database" )
    kvp = struct_gen.get_kvp( atoms, kvp, conc[0], conc[1] )
    struct_gen.setting.db.write(atoms,kvp)

def generate_exclusion_criteria( fname ):
    """
    Generate a selection critieria such that ids listed in the file given
    gets excluded
    """

    with open( fname, 'r' ) as infile:
        lines = infile.readlines()
    lines = [line.rstrip() for line in lines]
    ids = [int(line) for line in lines]
    select_cond = []
    for uid in ids:
        select_cond.append( ("id","!=",uid) )
    return select_cond

def enthalpy_of_formation( ceBulk, mode="compare" ):
    dft_energy = []
    ce_energy = []
    all_mg_conc = []
    dft_mg_conc = []
    dft_ids = []
    db = connect( db_name )
    al_ref_energy = None
    mg_ref_energy = None
    allowed_modes = ["compare", "CEGS"]
    with open( "data/almg_eci.json", 'r' ) as infile:
        ecis = json.load(infile)
    # Find reference structures
    for row in db.select(converged=1):
        dft_ids.append( row.id )
        if ( row.formula == "Al64" ):
            try:
                al_ref_energy = row.energy/64.0
            except:
                pass
        elif ( row.formula == "Mg64" ):
            mg_ref_energy = row.energy/64.0

    ce_ase_calc = ClusterExpansion( ceBulk, cluster_name_eci=ecis )
    atoms = bulk("Al")*(4,4,4)
    atoms.set_calculator(ce_ase_calc)
    ce_al_ref_energy = atoms.get_potential_energy()/64.0
    for i in range(64):
        atoms[i].symbol = "Mg"
    ce_mg_ref_energy = atoms.get_potential_energy()/64.0

    print ("Reference energies: Al: {}, Mg: {}".format( al_ref_energy, mg_ref_energy) )
    print ("CE Reference energies: Al: {}, Mg: {}".format( ce_al_ref_energy, ce_mg_ref_energy) )
    for row in db.select(converged=1):
        count = row.count_atoms()
        if ( "Mg" in count.keys() ):
            mg_conc = count["Mg"]/64.0
        else:
            mg_conc = 0.0

        dft_form = row.energy/64.0 - al_ref_energy*(1.0-mg_conc) - mg_ref_energy*mg_conc
        n_mg = int( mg_conc*len(ceBulk.atoms) )
        if( mg_conc in all_mg_conc ):
            dft_energy.append( dft_form )
            dft_mg_conc.append( mg_conc )
            if ( mode == "compare" ):
                atoms = db.get_atoms( id=row.id )
                min_energy = atoms.get_potential_energy()
                ce_form = min_energy/64.0 - al_ref_energy*(1.0-mg_conc) - mg_ref_energy*mg_conc
                ce_energy.append( ce_form )
                all_mg_conc.append( mg_conc )
                continue
        if ( (n_mg > 0) and (n_mg < 64) ):
            if ( mode == "CEGS" ):
                min_energy = find_gs_structure( ceBulk, mg_conc )
            else:
                atoms = db.get_atoms( id=row.id )
                min_energy = atoms.get_potential_energy()
            ce_form = min_energy/64.0 - al_ref_energy*(1.0-mg_conc) - mg_ref_energy*mg_conc
            ce_energy.append( ce_form )
            all_mg_conc.append( mg_conc )
            dft_energy.append( dft_form )
            dft_mg_conc.append( mg_conc )
        else:
            dft_energy.append( dft_form )
            dft_mg_conc.append( mg_conc )

        # Reset the atom in ceBulk
        for i in range( len(ceBulk.atoms) ):
            ceBulk.atoms[i].symbol = "Al"

    # Read ECIs
    with open("data/almg_eci.json", 'r') as infile:
        eci_data = json.load(infile)

    formation_res = {}
    formation_res["eci"] = eci_data
    formation_res["mg_conc"] = all_mg_conc
    formation_res["ce_formation"] = ce_energy
    formation_res["dft_formation"] = dft_energy
    formation_res["dft_mg_conc"] = dft_mg_conc
    formation_res["dft_ids"] = dft_ids
    outfilename = "data/almg_formation_energy.json"
    with open( outfilename, 'w') as outfile:
        json.dump( formation_res, outfile, indent=2, sort_keys=True, separators=(",",":") )
    print ( "Formation enthalpies written to {}".format( outfilename) )

def find_gs_structure( ceBulk, mg_conc ):
    fname = "data/{}.json".format(db_name.split(".")[0])
    with open( fname, 'r' ) as infile:
        ecis = json.load(infile)
    calc = CE( ceBulk, ecis )
    ceBulk.atoms.set_calculator( calc )
    conc = {
        "Mg":mg_conc,
        "Al":1.0-mg_conc
    }
    calc.set_composition(conc)
    print ( ceBulk.basis_functions )
    formula = ceBulk.atoms.get_chemical_formula()
    temps = [800,700,500,300,200,100,50,20,19,18,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1]
    n_steps_per = 1000
    lowest_struct = mcobs.LowestEnergyStructure( calc, None )
    for T in temps:
        print ("Temperature {}".format(T) )
        mc_obj = mc.Montecarlo( ceBulk.atoms, T )
        lowest_struct.mc_obj = mc_obj
        mc_obj.attach( lowest_struct )
        mc_obj.runMC( steps=n_steps_per, verbose=False )
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
    return lowest_struct.lowest_energy


def evalCE(BC):
    db = connect(BC.db_name)
    for row in db.select(converged=1):
        print(row.id, row.c0)
    lambs = np.logspace(-7,-1,num=50)
    #fname = "data/exclude_set_1.txt"
    #scond = generate_exclusion_criteria( fname )
    print (lambs)
    cvs = []
    evaluator = Evaluate(BC, fitting_scheme="l1")
    best_alpha = evaluator.plot_CV(alpha_min=1E-7, alpha_max=1E-1, num_alpha=50)
    evaluator.set_fitting_scheme(fitting_scheme="l1", alpha=best_alpha)
    eci_name = evaluator.get_cluster_name_eci()
    evaluator.plot_fit()
    plotter = ECIPlotter(eci_name)
    plotter.plot( show_names=True )

    eci_fname = "data/{}.json".format(db_name.split(".")[0])
    with open( eci_fname, 'w') as outfile:
        json.dump( eci_name, outfile )
    print ("ECIs stored in %s"%(eci_fname))
    plt.show()
    
def eval_phonons( ceBulk ):
    lambs = np.logspace(-2, 3,num=50)
    cnames = None
    cvs = []
    T = 600 # Is not used when hte=True
    for i in range(len(lambs)):
        print ("%d of %d"%(i,len(lambs)) )
        pce = PhononEval( ceBulk, lamb=lambs[i], penalty="l1", cluster_names=cnames, hte=True )
        pce.T = T
        cvs.append(pce._cv_loo() )
    indx = np.argmin(cvs)
    l = lambs[indx]
    pce = PhononEval( ceBulk, lamb=l, penalty="l1", cluster_names=cnames, filters=filters )
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
