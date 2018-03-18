import sys
sys.path.insert(1,"/home/dkleiven/Documents/ase/ase/spacegroup")
from ase.build import bulk
from ase.atoms import Atoms
from ase.visualize import view
from ase.build import bulk
import spacegroup as sp
get_spacegroup = sp.get_spacegroup
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

def prebeta_spacegroup():
    atoms = bulk("Al",cubic=True)

    # Add one site at the center
    L = atoms.get_cell()[0,0]
    at = Atoms( "Al", positions=[[L/2,L/2,L/2]] )
    atoms.extend(at)
    view(atoms)
    sp_gr = get_spacegroup(atoms)
    print (sp_gr)

def main():
    #prebeta()
    prebeta_spacegroup()

if __name__ == "__main__":
    main()
