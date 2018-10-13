import sys
sys.path.insert(2,"/home/davidkl/Documents/aseJin")
from ase.build import bulk
from ase.atoms import Atoms
from ase.visualize import view
from ase.build import bulk
from ase.spacegroup import get_spacegroup
from cemc.mcmc import CollectiveJumpMove, FixedElement
from cemc.tools import GSFinder
from ase.clease import CECrystal as BulkSpacegrou
from ase.clease import GenerateStructures
from ase.ce import Evaluate, CorrFunction
from numpy.random import randint, rand
from random import shuffle
from ase.io import write, read
from atomtools.ce import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
from ase.db import connect
plt.switch_backend("TkAgg")
import json
#from ase.spacegroup import *

L = 4.05
orig_positions = [(0,0,0),
                  (L,0,0.0),
                  (L,L,0.0),
                  (L,0,L),
                  (0.0,L,0.0),
                  (0.0,L,L),
                  (0.0,0.0,L),
                  (L,L,L),

                  (L/2,L/2,0),
                  (0,L/2,L/2),
                  (L/2,0,L/2),
                  (L,L/2,L/2),
                  (L/2,L,L/2),
                  (L/2,L/2,L)]


si = ["Si" for _ in range(6)]
mg = ["Mg" for _ in range(8)]
symbols = mg+si
eci_fname = "data/pre_beta_eci.json"
def prebeta(type="orig"):
    orig = Atoms(symbols=symbols,positions=orig_positions,cell=[[L,0,0],[0,L,0],[0,0,L]])
    view(orig)

def prebeta_spacegroup(n_template_structs=0):
    atoms = bulk("Al", cubic=True)

    # Add one site at the center
    L = atoms.get_cell()[0,0]
    at = Atoms( "X", positions=[[L/2,L/2,L/2]] )
    atoms.extend(at)
    view(atoms)
    sp_gr = get_spacegroup(atoms)
    print(sp_gr)
    print(atoms.get_scaled_positions())

    sc = atoms*(6,6,6)
    jump_moves = CollectiveJumpMove(mc_cell=sc)
    jump_moves.view_columns()
    temp_atoms = atoms*(2,2,2)
    print(len(temp_atoms))
    view(temp_atoms)

    if n_template_structs > 0:
        symbs = ["Al","Mg","Si"]
        for i in range(n_template_structs):
            atoms = bulk("Al",cubic=True,a=4.05)
            selection = [symbs[randint(low=0,high=3)] for _ in range(len(atoms))]
            for indx in range(len(atoms)):
                atoms[indx].symbol = selection[indx]
            at = Atoms( "X", positions=[[L/2,L/2,L/2]] )
            atoms.extend(at)
            atoms = atoms*(2,2,2)
            fname = "data/prebeta_template{}.xyz".format(i)
            write(fname,atoms)
            print("Template structure written to {}".format(fname))

db_name = "pre_beta.db"
def prebeta_ce(options):
    conc_args = {
        "conc_ratio_min_1":[[2,0,0,3]],
        "conc_ratio_max_1":[[0,1,1,3]],
        "conc_ratio_min_2":[[0,2,0,3]],
        "conc_ratio_max_2":[[1,0,1,3]]
    }
    a = 4.05
    basis = [(0,0,0),(0.5,0.5,0),(0.5,0.5,0.5)]
    cellpar = [a,a,a,90,90,90]
    basis_elements = [["Al","Mg","Si","X"],["Al","Mg","Si","X"],["Al","Mg","Si","X"]]
    bs = BulkSpacegroup(basis_elements=basis_elements, cellpar=cellpar, spacegroup=221, basis=basis,\
    db_name=db_name, max_cluster_size=4, size=[1, 1, 1], grouped_basis=[[0,1,2]], conc_args=conc_args,
    max_cluster_dia=4.048)
    print(bs.max_cluster_dia)

    # bs.reconfigure_settings()
    # view(bs.atoms_with_given_dim)
    # exit()
    # cf = CorrFunction(bs, parallel=True)
    # cf.reconfig_db_entries()
    # exit()

    struct_gen = GenerateStructures(bs,struct_per_gen=10)
    action = options[0]
    if action == "insert":
        fname = options[1]
        struct_gen.insert_structure(init_struct=fname)
    elif action == "rand_struct":
        insert_random_struct(bs, struct_gen,n_structs=10)
    elif action == "small_rand_struct":
        for i in range(100):
            insert_small_rand_struct(bs, struct_gen)

    elif action == "eval":
        evaluate(bs)
    elif action == "gs":
        atoms = read("data/template_jump.xyz")
        bs.atoms = atoms
        insert_gs_struct(bs, struct_gen, n_structs=30)

def insert_small_rand_struct(bs, gen):
    assert len(bs.atoms_with_given_dim) == 5
    symbs = ["X", "Al", "Al", "Al", "Al"]
    possible_symbs = ["Al", "Mg", "Si"]
    from random import choice
    for i in range(1, 5):
        symbs[i] = choice(possible_symbs)
    shuffle(symbs)
    atoms = bs.atoms_with_given_dim
    for i, atom in enumerate(atoms):
        atom.symbol = symbs[i]
    try:
        gen.insert_structure(init_struct=atoms)
    except Exception as exc:
        print(str(exc))

def insert_gs_struct(bs,struct_gen,n_structs=20):
    n_X = 16
    N = float(len(bs.atoms))
    assert len(bs.atoms) == 40
    gs_searcher = GSFinder()
    # fixed_vac = FixedElement(element="X")
    # gs_searcher.add_constraint(fixed_vac)
    counter = 0
    with open(eci_fname,'r') as infile:
        ecis = json.load(infile)

    for _ in range(n_structs):
        natoms = len(bs.atoms)-n_X
        max_Si = 0.5

        n_si = 0
        for atom in bs.atoms:
            if atom.symbol == "X":
                continue
            val = rand()
            if val < 0.05:
                atom.symbol = "Al"
            elif val < 0.7:
                atom.symbol = "Mg"
            else:
                atom.symbol = "Si"
                n_si += 1
            if ( n_si/N > 0.5 ):
                continue
        T = np.linspace(50,2000,20)[::-1]
        result = gs_searcher.get_gs(bs, ecis, temps=T)
        fname = "data/gs_serach.xyz"
        write(fname, result["atoms"])
        print ("Lowest energy: {} eV".format(result["energy"]/natoms))
        try:
            struct_gen.insert_structure( init_struct=fname )
            counter += 1
        except Exception as exc:
            print (str(exc))
    print ("Insert {} ground state strutures".format(counter))


def insert_random_struct(bs,struct_gen,n_structs=10):
    n_X = 16

    assert len(bs.atoms) == 40

    for _ in range(n_structs):
        natoms = len(bs.atoms)-n_X
        max_Si = 0.5
        n_al = randint(low=15,high=natoms)
        natoms -= n_al
        n_mg = randint(low=0,high=natoms)

        symbs = ["X"]*n_X
        symbs += ["Al"]*n_al
        symbs += ["Mg"]*n_mg
        while (len(symbs) < len(bs.atoms)):
            symbs.append("Si")
        shuffle(symbs)
        for i in range(len(bs.atoms)):
            bs.atoms[i].symbol = symbs[i]
        fname = "data/random_prebeta_structs.xyz"
        write(fname, bs.atoms)
        struct_gen.insert_structure(init_struct=fname)


def angles_filter(min_angle):
    db = connect(db_name)
    scond = []
    for row in db.select(converged=1):
        name = row.name
        atoms = row.toatoms()
        for struct in db.select(name=name):
            if struct.get("state", default="initial") == "relaxed":
                atoms = struct.toatoms()
        angles = atoms.get_cell_lengths_and_angles()[3:]
        if np.any(angles < min_angle):
            scond.append(("id", "!=", row.id))
    return scond


def evaluate(bs):
    from atomtools.ce import FilterCollapsed
    scond = FilterCollapsed.get_selection_condition(
        "data/filter_collapsed.db", 4000.0)
    scond += [("name", "!=", "conc_1.000_1.000_164")]
    print("Select condition: {}".format(scond))
    evaluator = Evaluate(bs, fitting_scheme"l2", parallel=True, select_cond=scond,
                         max_cluster_size=4)
    print(evaluate.get_cluster_correlation())
    exit()
    best_alpha = evaluator.plot_CV(alpha_min=1E-5, alpha_max=1E-2, num_alpha=16)
    evaluator.set_fitting_scheme(fitting_scheme="l2", alpha=best_alpha)
    evaluator.plot_fit(interactive=True)
    eci_name = evaluator.get_cluster_name_eci(return_type="dict")
    # classifier = FilterCollapsed(evaluator, db_name="data/filter_collapsed.db",
    #                              restart=False)
    # classifier.run(best_alpha, n_points=10, update_alpha=False)
    plotter = ECIPlotter(eci_name)
    plotter.plot()
    evaluator.plot_ECI()
    plt.show()

    with open(eci_fname,'w') as outfile:
        json.dump( eci_name, outfile, indent=2, separators=(",",":"))
    print ( "ECIs written to {}".format(eci_fname))

def main():
    #prebeta()
    #prebeta_spacegroup(n_template_structs=10)
    prebeta_ce(sys.argv[1:])

if __name__ == "__main__":
    main()
