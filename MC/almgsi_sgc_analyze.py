import dataset
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt

db_name = "sqlite:///data/almgsi_sgc_roi2.db"
def add_arrows(ax, x, y, step=10):
    for i in range(1,len(x)-1,step):
        ax.arrow( x[i], y[i], x[i]-x[i-1], y[i]-y[i-1], shape="full", lw=0, length_includes_head=True, head_width=0.01)
    return ax
def cooling_trajectories():
    db = dataset.connect(db_name)
    entries = db["systems"].find(status="finished")

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for entry in entries:
        sysID = entry["id"]
        data = db["thermodynamic"].find(sysID=sysID)
        c10 = []
        c11 = []
        T = []
        for entry in data:
            c10.append(entry["singlet_c1_0"])
            c11.append(entry["singlet_c1_1"])
            T.append(entry["temperature"])
        srt_indx = np.argsort(T)
        c10 = [c10[indx] for indx in srt_indx]
        c11 = [c11[indx] for indx in srt_indx]
        T = [T[indx] for indx in srt_indx]
        ax.plot( c10, c11, color="grey")
        ax.scatter(c10, c11, c=T, cmap="copper", marker="v")
    ax.set_xlabel("\$c_{10}\$")
    ax.set_ylabel("\$c_{11}\$")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

def main():
    cooling_trajectories()

if __name__ == "__main__":
    main()
