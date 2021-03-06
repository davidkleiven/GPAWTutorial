import sys
# sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
# sys.path.insert(2,"/home/dkleiven/Documents/aseJin")
# sys.path.insert(1,"/home/davidkl/Documents/aseJin")
# sys.path.append("/home/davidkl/Documents/GPAWTutorial/CE_extensions")
# sys.path.append("/home/dkleiven/Documents/GPAWTutorials/CE_extensions")
from ase.build import bulk
from clease import CEBulk as BulkCrystal
from clease import CorrFunction
from clease import NewStructures as GenerateStructures
# from atomtools.ce.corrmatrix import CovariancePlot
#from convex_hull_plotter import QHull
from clease.evaluate import Evaluate
# from atomtools.ce import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
#from cemc import CE
#from cemc.mcmc import mc_observers as mcobs
import json
#from cemc.mcmc import montecarlo as mc
from ase.io import write, read
from clease.calculator import Clease as ClusterExpansion
from ase.db import connect
from ase.units import mol, kJ
# from atomtools.ce import CVScoreHistory
from scipy.spatial import ConvexHull
# from atomtools.ce import ChemicalPotentialEstimator
from ase.calculators.singlepoint import SinglePointCalculator
from ase.visualize import view
from clease import Concentration

eci_fname = "data/almgsi_fcc_eci_newconfig.json"
#db_name = "almgsi_newconfig.db"
#db_name = "almgsi_multiple_templates.db"
#db_name = "almgsi_multiple_templates_dec10.db"
#db_name = "almgsi_sluiter.db"
#eci_fname = "data/almgsi_fcc_eci_sluiter.json"
# db_name_cubic = "almgsi_cubic.db"
db_name = "almgsi.db"

conc_fit = Concentration(basis_elements=[["Al", "Mg", "Si"]])
conc_fit.set_conc_ranges([[(0, 1), (0, 1), (0, 0.55)]])


def main(argv):
    option = argv[0]
    atoms = bulk("Al")
    N = 4
    atoms = atoms*(N, N, N)
    conc = Concentration(basis_elements=[["Al","Mg","Si"]])
    kwargs = dict(crystalstructure="fcc", a=4.05, size=[N, N, N], \
        db_name=db_name, max_cluster_size=4, concentration=conc,
        basis_function="sanchez", max_cluster_dia=[7.8, 5.0, 5.0])
    ceBulk = BulkCrystal(**kwargs)
    #ceBulk.reconfigure_settings()
    #print (ceBulk.basis_functions)
    #cf = CorrFunction( ceBulk, parallel=True)
    #cf.reconfigure_db_entries(select_cond=[("converged","=","1")])
    #exit()
    #print(ceBulk.basis_functions)
    struc_generator = GenerateStructures( ceBulk, struct_per_gen=10 )
    if ( option == "generateNew" ):
        struc_generator.generate_probe_structure()
    elif ( option == "eval" ):
        #evaluate(ceBulk)
        #evaluate_l1(ceBulk)
        evaluate_test(ceBulk)
        #evaluate_car(ceBulk)
        #evaluate_sklearn(ceBulk)
    elif option == "skew":
        insert_skewed_full(struc_generator, size1=int(argv[1]), size2=int(argv[2]))
    elif ( option == "insert" ):
        fname = argv[1]
        atoms = read(fname)
        struc_generator.insert_structure( init_struct=atoms )
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
    elif( option == "chempot" ):
        estimate_chemical_potentials(ceBulk)
    elif (option == "convert2cubic"):
        convert_to_cubic(ceBulk)
    elif option == "new_db":
        new_db_name = argv[1]
        new_db(kwargs, new_db_name)

def new_db(kwargs, new_db_name):
    db_name = "almgsi_newconfig.db"
    kwargs["db_name"] = new_db_name
    bc = BulkCrystal(**kwargs)
    names = []
    db = connect(db_name)
    for row in db.select(converged=1):
        names.append(row.name)

    ns = GenerateStructures(bc, struct_per_gen=10)
    for name in names:
        print(name)
        init_struct = None
        final_struct = None
        energy = 0.0
        for row in db.select(name=name):
            if row["calculator"] == "unknown":
                energy = row.energy
                init_struct = row.toatoms()
            else:
                final_struct = row.toatoms()
        calc = SinglePointCalculator(final_struct, energy=energy)
        final_struct.set_calculator(calc)
        try:
            ns.insert_structure(init_struct=init_struct, final_struct=final_struct, generate_template=True)
        except Exception as exc:
            print(str(exc))

def insert_skewed_full(struct_gen, size1=3, size2=1, max_number=200):
    from itertools import product
    symbs = ["Al", "Mg", "Si"]
    count = 0
    for s in product(symbs, repeat=size1*size2*2):
        if count > max_number:
            return
        atoms = bulk("Al", a=4.05)*(2, size1, size2)
        num_si = sum(1 for x in s if x == "Si")
        num_mg = sum(1 for x in s if x == "Mg")
        num_al = sum(1 for x in s if x == "Al")
        if num_al % size1 != 0:
            continue
        if num_si > 0.5*len(atoms):
            continue
        elif (abs(num_si - num_mg) != 0) and (num_si != 0) and(num_mg != 0):
            continue
        for i, atom in enumerate(atoms):
            atom.symbol = s[i]
        try:
            struct_gen.insert_structure(init_struct=atoms, generate_template=True)
            count += 1
        except:
            pass

def convert_to_cubic(setting_prim):
    conc_args = {
        "conc_ratio_min_1":[[64,0,0]],
        "conc_ratio_max_1":[[24,40,0]],
        "conc_ratio_min_2":[[64,0,0]],
        "conc_ratio_max_2":[[22,21,21]]
    }
    setting_cubic = BulkCrystal( crystalstructure="fcc", a=4.05, size=[3,3,3], basis_elements=[["Mg","Si","Al",]], \
    conc_args=conc_args, db_name=db_name_cubic, max_cluster_size=4, cubic=True )
    view(setting_cubic.atoms)
    atoms = setting_prim.atoms.copy()
    a = 4.05
    atoms.set_cell([[4*a,0,0],[0,4*a,0],[0,0,4*a]])
    atoms.wrap()
    view(atoms)
    print (setting_prim.atoms.get_cell())
    exit()
    out_file = "data/temp_out.xyz"
    target_cell = setting_cubic.atoms.get_cell()
    cubic_str_gen = struc_generator = GenerateStructures( setting_cubic, struct_per_gen=10 )
    db = connect(db_name)
    for row in db.select(converged=1):
        energy = row.energy
        atoms = row.toatoms()
        atoms.set_cell(target_cell)
        atoms.wrap()
        write(out_file,atoms)
        calc = SinglePointCalculator(atoms,energy=energy)
        atoms.set_calculator(calc)
        cubic_str_gen.insert_structure(init_struct=out_file,final_struct=atoms)

def update_in_conc_range():
    db = connect( db_name )
    for row in db.select():
        at_count = row.count_atoms()
        conc_si = 0.0
        if ( "Si" in at_count.keys() ):
            conc_si = at_count["Si"]/float(row.natoms)
        if ( conc_si > 0.32 ):
            db.update( row.id, in_conc_range=0 )

def evaluate_test(BC):
    from clease import BayesianCompressiveSensing

    scheme = BayesianCompressiveSensing(output_rate_sec=0.1, shape_var=0.5, noise=0.006, lamb_opt_start=20, variance_opt_start=15)
    evaluator = Evaluate(BC, fitting_scheme=scheme, parallel=False, alpha=1E-8,
                         scoring_scheme="loocv_fast", select_cond=[("converged", "=", True), ("calculator", "!=", "gpaw"), ("c1_1", "<", 0.5)])
    #evaluator.plot_fit(interactive=False, savefig=True, fname="data/bcs.png")
    X = evaluator.cf_matrix
    e_dft = evaluator.e_dft
    data = np.hstack((X, e_dft))
    np.savetxt("data/almgsi_X_e_dft.csv", delimiter=",", header=','.join(evaluator.cf_names) + ',e_dft')
    exit()
    evaluator.plot_fit(interactive=True)
    eci_name = evaluator.get_cluster_name_eci(return_type="dict")
    eci_bayes = "data/eci_bcs.json"
    with open(eci_bayes,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_bayes))

def evaluate_sklearn(BC):
    from ase.clease import ScikitLearnRegressor
    from sklearn.linear_model import LassoLars
    from sklearn.linear_model import OrthogonalMatchingPursuit
    from sklearn.linear_model import ARDRegression
    from sklearn.linear_model import BayesianRidge
    from sklearn.linear_model import HuberRegressor
    sk = LassoLars(alpha=1E-5, fit_intercept=False)
    sk = OrthogonalMatchingPursuit(n_nonzero_coefs=50, fit_intercept=False)
    sk = ARDRegression(fit_intercept=False)
    #sk = BayesianRidge(fit_intercept=False, alpha_1=1E-3, alpha_2=1E-3, lambda_1=1E-3, lambda_2=1E-3)
    #sk = HuberRegressor(fit_intercept=False, alpha=1E-8)
    scheme = ScikitLearnRegressor(sk)

    evaluator = Evaluate(BC, fitting_scheme=scheme, parallel=False, alpha=1E-8,
                         scoring_scheme="loocv", select_cond=[("converged", "=", True), ("calculator", "!=", "gpaw"), ("c1_1", "<", 0.5)])
    #evaluator.plot_fit(interactive=False, savefig=True, fname="data/bcs.png")
    evaluator.plot_fit(interactive=True)
    eci_name = evaluator.get_cluster_name_eci(return_type="dict")
    eci_bayes = "data/eci_scikit.json"
    with open(eci_bayes,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_bayes))

def evaluate_car(BC):
    from ase.clease import CarRank
    scheme = CarRank(car_percentile=75)
    evaluator = Evaluate(BC, fitting_scheme=scheme, parallel=False, alpha=1E-8,
                         scoring_scheme="loocv_fast", select_cond=[("converged", "=", True), ("calculator", "!=", "gpaw")])
    evaluator.plot_fit(interactive=True)
    eci_name = evaluator.get_cluster_name_eci(return_type="dict")
    eci_bayes = "data/eci_car.json"
    with open(eci_bayes,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_bayes))

def evaluate_l1(BC):
    from ase.clease import BayesianCompressiveSensing
    #from ase.clease import RandomValidator, EvenlyDistributedValidator
    #validator = EvenlyDistributedValidator(num_pca=3, num_hold_out=30, num_buckets=4)
    #validator = RandomValidator(num_hold_out=30)
    evaluator = Evaluate(BC, fitting_scheme='l1', parallel=False, alpha=1E-6,
                         select_cond=[("converged", "=", True), ("calculator", "!=", "gpaw")], scoring_scheme="k-fold")
    alpha = evaluator.plot_CV()
    evaluator.set_fitting_scheme(fitting_scheme="l1", alpha=alpha)
    evaluator.plot_fit(interactive=True)
    eci_name = evaluator.get_cluster_name_eci(return_type="dict")
    eci_bayes = "data/eci_l1.json"
    #evaluator.plot_CV()
    with open(eci_bayes,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_bayes))


def evaluate(BC):
    from ase.clease import GAFit
    cfunc = "loocv"

    ga_params = {"mutation_prob": 0.3,
                 "num_individuals": 100,
                 "fname": "data/ga_fit_almgsi_{}.csv".format(cfunc),
                 "select_cond": [("converged", "=", True), ("calculator", "!=", "gpaw"), ("c1_1", "<", 0.5)],
                 "cost_func": cfunc,
                 "sparsity_slope": 1.0,
                 "min_weight": 1.0,
                 "include_subclusters": False}
                 #"conc_constraint": conc_fit}
    # ga = GAFit(setting=BC, alpha=1E-8, mutation_prob=0.1, num_individuals="auto",
    #            change_prob=0.2, fname="data/ga_fit_almgsi_{}.csv".format(cfunc), 
    #            select_cond=[("converged", "=", True)],
    #            cost_func=cfunc, sparsity_slope=1.5, min_weight=1E-1)

    ga = GAFit(setting=BC, **ga_params)
    #names = ga.run(min_change=0.001, gen_without_change=10000, save_interval=1000)
    #exit()
    name_file = "data/ga_fit_almgsi_{}_cluster_names.txt".format(cfunc)
    with open(name_file, 'r') as infile:
        lines = infile.readlines()
    names = [x.strip() for x in lines]

    eval_params = dict(parallel=False, alpha=1E-8,
                         scoring_scheme="k-fold", 
                         select_cond=[("converged", "=", True)], 
                         min_weight=ga_params["min_weight"], fitting_scheme="l2", cluster_names=names,
                         num_repetitions=100, nsplits=10)
    evaluator = Evaluate(BC, **eval_params)
    #evaluator.plot_CV()
    #exit()

    evaluator.plot_fit(interactive=True)
    eci_name = evaluator.get_cluster_name_eci(return_type="dict")
    eci_fname_ga = "eci_almgsi_{}.json".format(cfunc)

    output = {}
    ga_params.pop("conc_constraint")
    eval_params.pop("conc_constraint")
    output["eci"] = eci_name
    output["ga_params"] = ga_params
    output["eval_params"] = eval_params
    output["rmse"] = evaluator.rmse()
    output["loocv"] = evaluator.loocv_fast()
    with open(eci_fname_ga,'w') as outfile:
        json.dump( output, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_fname_ga))

def estimate_chemical_potentials(ceBulk):
    c1_0 = []
    c1_1 = []
    F = []
    db = connect( ceBulk.db_name )
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
        F.append(row.energy/row.natoms)
        c1_0.append( row["c1_0"] )
        c1_1.append( row["c1_1"] )
    singl = np.vstack((c1_0,c1_1) ).T
    estimator = ChemicalPotentialEstimator( singlets=singl, energies=F )
    estimator.plot()
    mu0 = []
    mu1 = []
    for i in range(len(c1_0)):
        x = np.array( [c1_0[i],c1_1[i]] )
        mu0.append( estimator.deriv( x, 0 ) )
        mu1.append( estimator.deriv( x, 1 ) )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( c1_0, mu0, "o", mfc="none", label="mu0" )
    ax.plot( c1_1, mu1, "o", mfc="none", label="mu1" )
    ax.legend()
    plt.show()

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
    srt_indx = np.argsort(conc_mg)
    conc_mg = np.array( [conc_mg[indx] for indx in srt_indx] )
    conc_si = np.array( [conc_si[indx] for indx in srt_indx] )
    enthalpy_dft = np.array( [enthalpy_dft[indx] for indx in srt_indx] )
    enthalpy_ce = np.array( [enthalpy_ce[indx] for indx in srt_indx] )
    formulas = [formulas[indx] for indx in srt_indx]
    enthalpy_dft = np.array(enthalpy_dft)*mol/kJ
    enthalpy_ce = np.array(enthalpy_ce)*mol/kJ
    cm = plt.cm.get_cmap('RdYlBu')
    vmin = np.min(conc_si)
    vmax = np.max(conc_si)
    ax.plot( conc_mg, enthalpy_dft, "o", mfc="none", color="#bdbdbd"  )
    im = ax.scatter( conc_mg, enthalpy_ce, s=30, marker="x", c=conc_si, cmap=cm, vmin=vmin,vmax=vmax )

    qhull_color = "#bdbdbd"
    hull = ConvexHull( np.vstack((conc_mg, enthalpy_dft)).T )
    for simplex in hull.simplices:
        E1 = enthalpy_dft[simplex[0]]
        E2 = enthalpy_dft[simplex[1]]
        x1 = conc_mg[simplex[0]]
        x2 = conc_mg[simplex[1]]
        ax.text( x1, E1, formulas[simplex[0]])
        ax.text( x2, E2, formulas[simplex[1]])
        if ( E1 <= 0.0 and E2 <= 0.0 ):
            ax.plot( [x1,x2], [E1,E2], color=qhull_color )

    ax.set_xlabel( "\$c_{Mg}/(c_{Mg}+c_{Si})")
    ax.set_ylabel( "Enthalpy of formation (kJ/mol)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    #for i in range(len(formulas)):
    #    ax.text( x[i], enthalpy_dft[i], formulas[i] )
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
