from ase.db import connect
import sys
from matplotlib import pyplot as plt
import numpy as np

def main(name):
    a = []
    energy = []
    db = connect("almgsiX_fcc.db")
    for row in db.select(name=name, calculator="gpaw"):
        a.append(row.lattice_param)
        energy.append(row.energy)

    print(a, energy)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(a, energy, "o", mfc="none")

    # Perform a least squares fit
    A = np.zeros((len(a), 3))
    A[:, 0] = 1.0
    A[:, 1] = a
    A[:, 2] = np.array(a)**2
    x, res, rank, s = np.linalg.lstsq(A, energy)
    a_fit = np.linspace(0.95*np.min(a), 1.1*np.max(a), 100)
    E_fit = x[0] + x[1]*a_fit + x[2]*a_fit**2
    ax.plot(a_fit, E_fit)
    a_opt = -0.5*x[1]/x[2]
    print("Optimal lattice parameter: {}".format(a_opt))
    # ax.plot([a_opt], [x[0] + x[1]*a_opt, x[2]*a_opt**2], "x")
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
