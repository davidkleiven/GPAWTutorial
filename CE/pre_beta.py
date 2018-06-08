import sys
sys.path.insert(2,"/home/davidkl/Documents/aseJin")
from ase.build import bulk
from ase.atoms import Atoms
from ase.visualize import view
from ase.build import bulk
from ase.spacegroup import get_spacegroup
from cemc.mcmc import CollectiveJumpMove
from cemc.tools import GSFinder
from ase.ce import BulkSpacegroup, GenerateStructures
from numpy.random import randint
from random import shuffle
from ase.io import write
from ase.ce.evaluate import Evaluate
from atomtools.ce import ECIPlotter
import numpy as np
from matplotlib import pyplot as plt
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
    atoms = bulk("Al",cubic=True)

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
    db_name=db_name,max_cluster_size=4, size=[2,2,2], grouped_basis=[[0,1,2]], conc_args=conc_args)

    struct_gen = GenerateStructures(bs,struct_per_gen=10)
    action = options[0]
    if action == "insert":
        fname = options[1]
        struct_gen.insert_structure(init_struct=fname)
    elif action == "rand_struct":
        insert_random_struct(bs, struct_gen,n_structs=10)
    elif action == "eval":
        evaluate(bs)
    elif action == "gs":
        insert_gs_struct(bs,struct_gen)

def insert_gs_struct(bs,struct_gen,n_structs=20):
    n_X = 8
    N = float(len(bs.atoms))
    assert len(bs.atoms) == 40
    gs_searcher = GSFinder()
    counter = 0
    with open(eci_fname,'r') as infile:
        ecis = json.load(infile)

    for _ in range(n_structs):
        natoms = len(bs.atoms)-n_X
        max_Si = 0.5
        n_al = randint(low=15,high=natoms)
        natoms -= n_al
        n_mg = randint(low=0,high=natoms)
        n_si = N-n_al-n_mg-n_X
        if ( n_si/N > 0.5 ):
            continue
        conc = {
            "X":8.0/N,
            "Al":n_al/N,
            "Mg":n_mg/N,
            "Si":n_si/N
        }
        T = np.linspace(50,2000,20)[::-1]
        result = gs_searcher.get_gs(bs, ecis, composition=conc, temps=T)
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
    n_X = 8

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
        write(fname,bs.atoms)
        struct_gen.insert_structure(init_struct=fname)

def evaluate(bs):
    lambs = np.logspace(-8,-4,num=50)
    cvs = []
    s_cond = []
    for i in range(len(lambs)):
        print (lambs[i])
        evaluator = Evaluate( bs, lamb=float(lambs[i]), penalty="l1", select_cond=s_cond )
        cvs.append(evaluator._cv_loo())
    indx = np.argmin(cvs)
    print ("Selected penalization: {}".format(lambs[indx]))
    evaluator = Evaluate( bs, lamb=float(lambs[indx]), penalty="l1", select_cond=s_cond )
    eci_name = evaluator.get_cluster_name_eci_dict
    evaluator.plot_energy()
    plotter = ECIPlotter(eci_name)
    plotter.plot()
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
