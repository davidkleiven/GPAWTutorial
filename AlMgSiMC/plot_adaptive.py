import sys
import h5py as h5
import matplotlib as mpl
import numpy as np
rc = {"axes.unicode_minus": False, "font.size": 18, "svg.fonttype": "none"}
mpl.rcParams.update(rc)

def plot(fname):
    from matplotlib import pyplot as plt
    with h5.File(fname, 'r') as hfile:
        x = np.array(hfile["x"])
        visits = np.array(hfile["visits"])
        bias = np.array(hfile["bias"])

    bias -= bias[0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, -bias, drawstyle="steps")
    ax.set_xlabel("Shape parameter, \$Q_2$\$")
    ax.set_ylabel("\$\Delta G/k_BT\$")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    plot(sys.argv[1])


    

