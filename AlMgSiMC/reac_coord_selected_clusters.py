from ase.io import read
import numpy as np

fnames = ["long_arms_cluster2.xyz", "long_arms_cluster1.xyz",
          "needle.xyz", "spherical_cluster.xyz"]

def calculate_reac_crd():
    for fname in fnames:
        atoms = read("data/" + fname)
        pos = atoms.get_positions()
        com = np.mean(pos, axis=0)
        pos -= com
        I = np.zeros((3, 3))
        for i in range(pos.shape[0]):
            I += np.outer(pos[i, :], pos[i, :])

        eigvals = np.sort(np.linalg.eigvals(I))
        Q = 1.0 - (eigvals[0] + eigvals[1])/(2*eigvals[2])
        print(fname, Q)

if __name__ == "__main__":
    calculate_reac_crd()