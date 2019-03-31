from ase.io import write, read
from ase.visualize import view
import numpy as np  


def main():
    fname = "data/clustered_mc_digi.xyz"

    atoms = read(fname)
    diag = 0.5*np.sum(atoms.get_cell(), axis=0)
    atoms.translate(diag - atoms[28539].position)
    atoms.wrap()
    view(atoms)
    write("data/cluster_mc_digi_low_temp.cif", atoms)


if __name__ == "__main__":
    main()