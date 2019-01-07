import sys
from ase.io.trajectory import Trajectory
import numpy as np
from copy import deepcopy
from ase.calculators.singlepoint import SinglePointCalculator

def extract_cluster(atoms):
    max_dist = 3.0
    mg_indx = [atom.index for atom in atoms if atom.symbol == "Mg"]
    indices = deepcopy(mg_indx)
    for indx in mg_indx:
        indx_dist = list(range(len(atoms)))
        del indx_dist[indx]
        indx_dist = np.array(indx_dist)
        dists = atoms.get_distances(indx, indx_dist, mic=True)
        dists = np.array(dists)
        new_indx = indx_dist[dists<max_dist]
        indices += list(new_indx)
    indices = list(set(indices))
    cluster = atoms[indices]
    cell = cluster.get_cell()
    diag = 0.5*(cell[0, :] + cell[1, :] + cell[2, :])
    indices = list(range(1, len(cluster)))
    dist = cluster.get_distances(0, indices, mic=True, vector=True)
    com = np.mean(dist, axis=0) + cluster[0].position
    cluster.translate(diag-com)
    cluster.wrap()

    energy = atoms.get_potential_energy()
    calc = SinglePointCalculator(cluster, energy=energy)
    cluster.set_calculator(calc)
    return cluster

def main(fname):
    fname_out = fname.split(".")[0]
    fname_out += "only_cluster.traj"
    traj = Trajectory(fname, mode="r")
    traj_clust = Trajectory(fname_out, mode="w")
    for i, atoms in enumerate(traj):
        print("{} of {}".format(i, len(traj)))
        cluster = extract_cluster(atoms)
        traj_clust.write(cluster)

if __name__ == "__main__":
    main(sys.argv[1])
