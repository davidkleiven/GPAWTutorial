import h5py as h5
from ase.units import kB
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({"axes.unicode_minus": False, "font.size": 18, "svg.fonttype": "none"})

folder = "data/diffraction"
temp = [600, 700, 800]


def main():
    from matplotlib import pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for T in temp:
        fname = folder + "/layered_bias{}K.h5".format(T)
        with h5.File(fname, 'r') as hf:
            x = np.array(hf["x"])
            G = -np.array(hf["bias"])/(kB*T)
        vol = 1000.0*4.05**3
        G /= vol
        G -= np.min(G)
        ax.plot(x, G, drawstyle="steps", label="{}K".format(T))
    ax.legend()
    ax.set_xlabel("\$| \sum_{Mg}\exp(ik_x x) |\$")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("\$\Delta G \SI{}{\electronvolt\per\\angstromg^3}\$")
    plt.show()

if __name__ == "__main__":
    main()
