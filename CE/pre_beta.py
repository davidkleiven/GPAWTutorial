import sys
sys.path.insert(2,"/home/davidkl/Documents/aseJin")
from ase.build import bulk
from ase.atoms import Atoms
from ase.visualize import view
from ase.build import bulk
from ase.spacegroup import get_spacegroup
from cemc.mcmc import CollectiveJumpMove
from ase.ce import BulkSpacegroup, GenerateStructures
from numpy.random import randint
from ase.io import write
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


def main():
    #prebeta()
    #prebeta_spacegroup(n_template_structs=10)
    prebeta_ce(sys.argv[1:])

if __name__ == "__main__":
    main()
