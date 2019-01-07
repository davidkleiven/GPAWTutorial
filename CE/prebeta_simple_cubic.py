import sys
from ase.clease import CEBulk as BulkCrystal
from ase.clease import NewStructures as GenerateStructures
from ase.clease import Concentration
from ase.build import bulk
from ase.clease import CorrFunction
from ase.clease import Evaluate
from ase.clease import GAFit
# from atomtools.ce import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
import json

db_name = "pre_beta_simple_cubic.db"
#db_name = "prebeta_sc.db"
#db_name = "prebeta_sc_large_cluster.db"

a = 2.025

eci_fname = "data/eci_pre_beta_simple_cubic.json"
def main(argv):
    option = argv[0]
    print(option)
    conc_args = {
        "conc_ratio_min_1": [[2, 0, 0, 3]],
        "conc_ratio_max_1": [[0, 1, 1, 3]],
        "conc_ratio_min_2": [[0, 2, 0, 3]],
        "conc_ratio_max_2": [[1, 0, 1, 3]]
    }
    a = 2.025
    conc = Concentration(basis_elements=[["Al", "Mg", "Si", "X"]],)
    kwargs = dict(crystalstructure="sc", size=[4, 4, 2],
                     max_cluster_size=4,
                     max_cluster_dia=[0, 0, 7.0, 5.0, 5.0],
                     a=a,
                     concentration=conc,
                     db_name=db_name)
    #bc = BulkCrystal(**kwargs)
    #reconfig(bc)
    #bc.reconfigure_settings()
    #exit()
    #struct_gen = GenerateStructures(bc, struct_per_gen=10)
    if option == "new_prebeta_random":
        gen_random_struct(bc, struct_gen, lattice="prebeta", n_structs=30)
    elif option == "new_fcc_random":
        gen_random_struct(bc, struct_gen, lattice="fcc", n_structs=20)
    elif option == "new_pre_beta_gs":
        gen_gs_prebeta(bc, struct_gen, n_structs=20)
    elif option == "new_fcc_gs":
        gen_gs_prebeta(bc, struct_gen, n_structs=10, lattice="fcc")
    elif option == "eval":
        evaluate(bc)
    elif option == "score":
        calculate_score()
    elif option == "filter":
        filter_atoms()
    elif option == "new_db":
        new_db_name = argv[1]
        create_brand_new_db(kwargs, new_db_name)

def create_brand_new_db(kwargs, db_name):
    from ase.calculators.singlepoint import SinglePointCalculator
    from ase.db import connect
    old_db_name = kwargs["db_name"]
    db = connect(old_db_name)
    kwargs["db_name"] = db_name
    bc = BulkCrystal(**kwargs)
    ns = GenerateStructures(bc)
    names = []
    for row in db.select(converged=1):
        names.append(row.name)
    
    for name in names:
        print(name)
        init_atoms = None
        final_atoms = None
        for row in db.select(name=name):
            energy = 0.0
            if row["calculator"] == "unknown":
                init_atoms = row.toatoms()
                energy = row.energy
            else:
                final_atoms = row.toatoms(attach_calculator=True)
                final_atoms.get_calculator().results["energy"] = row.energy
        #calc = SinglePointCalculator(final_atoms, energy=row.energy)
        #final_atoms.set_calculator(calc)
                
        
        ns.insert_structure(init_struct=init_atoms, final_struct=final_atoms, generate_template=True)

def calculate_score():
    from atomtools.ce import DistanceDistribution
    from ase.db import connect
    from ase.visualize import view
    db = connect(db_name)
    atoms = []
    names = []
    for row in db.select(converged=True):
        names.append(row.name)

    # Calculate the score
    ks_dists = []
    for name in names:
        show = False
        print("Current name: {}".format(name))
        atoms = []
        for row in db.select(name=name):
            atoms.append(row.toatoms())
        assert len(atoms) == 2
        init = atoms[0]
        final = atoms[1]
        if len(final) < 6 or len(init) < 6:
            continue
        dist1 = DistanceDistribution(init)
        dist2 = DistanceDistribution(final)
        ks_dist = dist1.kolmogorov_smirnov_distance(dist2, show=show)
        ks_dists.append(ks_dist)
    
    score_name = list(zip(ks_dists, names))
    score_name.sort()
    for item in score_name:
        print("{}: {}".format(item[1], item[0]))

def filter_atoms():
    from atomtools.ce import FilterDisplacements
    from ase.db import connect
    from ase.visualize import view
    from ase.io import Trajectory
    db = connect(db_name)
    names = []
    uids = []
    disp_filter = FilterDisplacements(max_displacement=1.5)
    for row in db.select(converged=True):
        names.append(row.name)
        uids.append(row.id)

    traj = Trajectory("data/prebeta_sc_valid.traj", mode="w")
    num_valid = 0
    for name, uid in zip(names, uids):
        atoms = []
        for row in db.select(name=name):
            atoms.append(row.toatoms())
        assert len(atoms) == 2
        valid = disp_filter.is_valid(atoms[0], atoms[1], num_attempts=10)
        if valid:
            traj.write(atoms[0])
            traj.write(atoms[1])
            num_valid += 1
            db.update(uid, is_valid=True)
        else:
            print(disp_filter.rejected_reason)
            db.update(uid, is_valid=False)
    print("Number of valid structures: {}".format(num_valid))

def reconfig(bc):
    #scond = [("converged", "=", True)]
    #bc.reconfigure_settings()
    cf = CorrFunction(bc, parallel=True)
    cf.reconfigure_db_entries()
    exit()


def gen_gs_prebeta(bc, struct_gen, n_structs=10, lattice="prebeta"):
    from cemc.tools import GSFinder
    from cemc.mcmc import FixedElement
    from ase.clease.tools import wrap_and_sort_by_position
    from random import choice
    from ase.build import cut
    from cemc import CE
    #constraint = FixedElement(element="X")
    symbs = ["Al", "Mg", "Si"]
    gs_searcher = GSFinder()
    #gs_searcher.add_constraint(constraint)

    with open(eci_fname, 'r') as infile:
        ecis = json.load(infile)
    from ase.visualize import view

    if lattice == "prebeta":
        s = bc.size
        scale_factor = [int(s[0]/4), int(s[1]/4), int(s[2]/2)]
        atoms = get_pre_beta_template() * scale_factor
        atoms = wrap_and_sort_by_position(atoms)
    else:
        s = bc.size
        scale_factor = [int(s[0]/2), int(s[1]/2), int(s[2]/2)]
        atoms = get_fcc_template() * scale_factor
        atoms = wrap_and_sort_by_position(atoms)

    # for i, atom in enumerate(atoms):
    #     bc.atoms[i].symbol = atom.symbol

    #inv_scale_factor = [1.0/factor for factor in bc.supercell_scale_factor]
    for i in range(n_structs):
        print("Generating {} of {} structures".format(i, n_structs))
        calc = CE(bc, ecis)
        bc.atoms.set_calculator(calc)
        symbols = [atom.symbol for atom in atoms]
        for i in range(len(symbols)):
            if symbols[i] == "X":
                continue
            symbols[i] = choice(symbs)
        calc.set_symbols(symbols)

        # Add new tags
        #atoms = bc.atoms.copy()
        # cell = atoms.get_cell()
        # #atoms = atoms * bc.supercell_scale_factor
        # atoms = wrap_and_sort_by_position(atoms)
        # for i, atom in enumerate(atoms):
        #     bc.atoms[i].symbol = atom.symbol
        #cell_large = bc.atoms.get_cell()
        temps = np.linspace(10.0, 1500.0, 30)[::-1]
        print(bc.atoms.get_chemical_formula())
        gs = gs_searcher.get_gs(bc, None, temps=temps, n_steps_per_temp=10 * len(bc.atoms))
        #atoms = cut(gs["atoms"], a=(inv_scale_factor[0], 0, 0), b=(0, inv_scale_factor[1], 0), c=(0, 0, inv_scale_factor[2]))
        try:
            struct_gen.insert_structure(init_struct=gs["atoms"])
        except Exception as exc:
            print(str(exc))

def get_fcc_template():
    from ase.clease.tools import wrap_and_sort_by_position
    atoms = bulk("Al", crystalstructure="sc", a=a)
    atoms = atoms*(2, 2, 2)

    vac_pos = [
        [a, 0, 0],
        [0, a, 0],
        [0, 0, a],
        [a, a, a]
    ]
    vac_pos = np.array(vac_pos)
    pos = atoms.get_positions()
    for i in range(vac_pos.shape[0]):
        lengths = np.zeros(pos.shape[0])
        for j in range(pos.shape[0]):
            diff = pos[j, :] - vac_pos[i, :]
            lengths[j] = np.sqrt(np.sum(diff**2))
        closest = np.argmin(lengths)
        atoms[closest].symbol = "X"
    return wrap_and_sort_by_position(atoms)


def get_pre_beta_template():
    from ase.clease.tools import wrap_and_sort_by_position
    atoms = get_fcc_template()
    atoms = atoms * (2, 2, 1)
    atoms = wrap_and_sort_by_position(atoms)
    atoms[10].symbol = "X"
    atoms[11].symbol = "Al"
    return atoms


def gen_random_struct(bc, struct_gen, n_structs=20, lattice="fcc"):
    candidate_symbs = ["Al", "Mg", "Si"]
    from random import choice
    for i in range(n_structs):
        print("Generating random structure: {}".format(i))
        if lattice == "fcc":
            template = get_fcc_template()
        elif lattice == "prebeta":
            template = get_pre_beta_template()
        else:
            raise ValueError("Unknown lattice type {}".format(lattice))
        for atom in template:
            if atom.symbol == "X":
                continue
            atom.symbol = choice(candidate_symbs)
        try:
            struct_gen.insert_structure(init_struct=template)
        except Exception as exc:
            print(str(exc))

def evaluate(bc):
    from ase.db import connect
    #from ase.clease import BranchAndBound
    
    scond = [("converged", "=", True)]
    reject_angles = True
    if reject_angles:
        with open("data/rejected_names_based_on_angles.json", 'r') as infile:
            res = json.load(infile)
        for name in res["names"]:
            scond.append(("name", "!=", name))
    #db = connect("pre_beta_simple_cubic.db")
    # for row in db.select(scond):
    #     value = row.get("c3_2p2864_8_100")
    #     if value is None:
    #         print(row.id)
    #         print(row.key_value_pairs)
    # exit()
    scheme = "l2"
    evaluator = Evaluate(bc, fitting_scheme=scheme, parallel=False,
                         max_cluster_size=4, scoring_scheme="loocv_fast",
                         select_cond=scond)
    # bnb = BranchAndBound(evaluator, max_num_eci=100, bnb_cost="aic")
    # bnb.select_model()

    ga = GAFit(evaluator=evaluator, alpha=1E-8, mutation_prob=0.01, num_individuals="auto",
               change_prob=0.2, fname="data/ga_fit_dec14.csv", parallel=False, max_num_in_init_pool=150)
    ga.run(min_change=0.001)
    #ga.plot_evolution()
    # exit()

    #best_alpha = evaluator.plot_CV(alpha_min=1E-5, alpha_max=1E-1, num_alpha=16)
    #np.savetxt("data/cfm_prebeta_sc.csv", evaluator.cf_matrix, delimiter=",")
    #np.savetxt("data/e_dft_prebeta_sc.csv", evaluator.e_dft)
    #evaluator.set_fitting_scheme(scheme, best_alpha)
    evaluator.plot_fit(interactive=False, savefig=True, fname="data/prebeta_ga.png")
    eci_name = evaluator.get_cluster_name_eci(return_type="dict")
    #plt.show()

    with open(eci_fname,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_fname))


if __name__ == "__main__":
    main(sys.argv[1:])
