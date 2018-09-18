from ase.io.trajectory import Trajectory
import sys


def main(fname):
    traj = Trajectory(fname, mode="r")
    fname_out = fname.rpartition(".")[0]+"_cluster.traj"
    traj_out = Trajectory(fname_out, mode="w")
    for i in range(len(traj)):
        atoms = traj[i]
        indices = [atom.index for atom in atoms if atom.symbol != "Al"]
        cluster = atoms[indices]
        indices = list(range(1, len(cluster)))
        dists = cluster.get_distances(0, indices, mic=True, vector=True)
        com = cluster[0].position + dists.sum(axis=0)/len(cluster)
        cell = cluster.get_cell()
        diag = 0.5 * (cell[:, 0] + cell[:, 1] + cell[:, 2])
        cluster.translate(diag-com)
        cluster.wrap()
        cluster.center()
        traj_out.write(cluster)
    print("Only clusters written to file {}".format(fname_out))


if __name__ == "__main__":
    main(sys.argv[1])
