import sys
from ase.clease import CEBulk as BulkCrystal
from ase.clease import GenerateStructures
from ase.build import bulk
from ase.clease import CorrFunction
from ase.clease import Evaluate
# from atomtools.ce import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
import json

db_name = "pre_beta_simple_cubic.db"

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
    bc = BulkCrystal(crystalstructure="sc", size=[8, 8, 2],
                     conc_args=conc_args, max_cluster_size=4,
                     max_cluster_dia=[0, 0, 5.0, 3.0, 3.0],
                     basis_elements=[["Al", "X", "Mg", "Si"]], a=a,
                     db_name=db_name)
    # reconfig(bc)
    # bc.reconfigure_settings()
    # exit()
    struct_gen = GenerateStructures(bc, struct_per_gen=10)
    if option == "new_prebeta_random":
        gen_random_struct(bc, struct_gen, lattice="prebeta", n_structs=30)
    elif option == "new_fcc_random":
        gen_random_struct(bc, struct_gen, lattice="fcc", n_structs=20)
    elif option == "new_pre_beta_gs":
        gen_gs_prebeta(bc, struct_gen, n_structs=9)
    elif option == "new_fcc_gs":
        gen_gs_prebeta(bc, struct_gen, n_structs=10, lattice="fcc")
    elif option == "eval":
        evaluate(bc)


def reconfig(bc):
    bc.reconfigure_settings()
    cf = CorrFunction(bc, parallel=True)
    cf.reconfig_db_entries()
    exit()


def gen_gs_prebeta(bc, struct_gen, n_structs=10, lattice="prebeta"):
    from cemc.tools import GSFinder
    from cemc.mcmc import FixedElement
    from ase.clease.tools import wrap_and_sort_by_position
    from random import choice
    from ase.build import cut
    constraint = FixedElement(element="X")
    symbs = ["Al", "Mg", "Si"]
    gs_searcher = GSFinder()
    gs_searcher.add_constraint(constraint)

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

    for i, atom in enumerate(atoms):
        bc.atoms_with_given_dim[i].symbol = atom.symbol

    inv_scale_factor = [1.0/factor for factor in bc.supercell_scale_factor]
    for i in range(n_structs):
        print("Generating {} of {} structures".format(i, n_structs))
        for atom in bc.atoms_with_given_dim:
            if atom.symbol == "X":
                continue
            atom.symbol = choice(symbs)

        # Add new tags
        atoms = bc.atoms_with_given_dim.copy()
        cell = atoms.get_cell()
        atoms = atoms * bc.supercell_scale_factor
        atoms = wrap_and_sort_by_position(atoms)
        for i, atom in enumerate(atoms):
            bc.atoms[i].symbol = atom.symbol
        cell_large = bc.atoms.get_cell()
        temps = np.linspace(10.0, 1500.0, 30)[::-1]
        gs = gs_searcher.get_gs(bc, ecis, temps=temps, n_steps_per_temp=10 * len(bc.atoms))
        atoms = cut(gs["atoms"], a=(inv_scale_factor[0], 0, 0), b=(0, inv_scale_factor[1], 0), c=(0, 0, inv_scale_factor[2]))
        try:
            struct_gen.insert_structure(init_struct=atoms)
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
    evaluator = Evaluate(bc, fitting_scheme="l2", parallel=False,
                         max_cluster_size=4, scoring_scheme="loocv_fast")

    best_alpha = evaluator.plot_CV(alpha_min=1E-3, alpha_max=1E-1, num_alpha=16)
    np.savetxt("data/cfm_prebeta_sc.csv", evaluator.cf_matrix, delimiter=",")
    np.savetxt("data/e_dft_prebeta_sc.csv", evaluator.e_dft)
    evaluator.set_fitting_scheme("l2", best_alpha)
    evaluator.plot_fit(interactive=True)
    eci_name = evaluator.get_cluster_name_eci(return_type="dict")
    plt.show()

    with open(eci_fname,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_fname))


if __name__ == "__main__":
    main(sys.argv[1:])
