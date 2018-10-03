from matplotlib import pyplot as plt
import dataset
import numpy as np

DB_NAME = "data/large_cluster/mc_surface.db"

def fit(T):
    db = dataset.connect("sqlite:///{}".format(DB_NAME))
    tab = db["mc_backup"]
    energy = []
    natoms = []
    al_conc = []
    for row in tab.find(temperature=T):
        energy.append(row["energy"])
        natoms.append(int(row["natoms"]))
        al_conc.append(row["Al_conc"])
    print(natoms)

    # Convert to numpy arrays
    energy = np.array(energy)
    natoms = np.array(natoms)
    al_conc = np.array(natoms)

    matrix = np.zeros((len(energy), 2))
    matrix[:, 0] = 1.0
    matrix[:, 1] = natoms
    x, res, rank, s = np.linalg.lstsq(matrix, energy)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(natoms, energy, "o")
    ax.plot(natoms, x[0] + x[1]*natoms)
    print("Intersection: {}".format(x[0]))
    print("Slope: {}".format(x[1]))
    plt.show()
if __name__ == "__main__":
    fit(400)
