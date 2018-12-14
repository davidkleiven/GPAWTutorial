import dataset
import matplotlib as mpl
mpl.rcParams.update({"font.size": 18, "axes.unicode_minus": False, "svg.fonttype": "none"})
from matplotlib import pyplot as plt
import numpy as np

def main():
    db = dataset.connect("sqlite:///data/mc_solute.db")
    tbl = db["mc_backup"]
    I00 = []
    I11 = []
    I22 = []
    for row in tbl.find(temperature=10):
        I = np.array([[row["I00"], row["I01"], row["I02"]],
                      [row["I10"], row["I11"], row["I12"]],
                      [row["I20"], row["I21"], row["I22"]]])
        eigs, eigvec = np.linalg.eigh(I)
        print(eigvec)
        eigs = eigs[::-1]
        I00.append(eigs[0])
        I11.append(eigs[1])
        I22.append(eigs[2])

    I00 = np.array(I00)
    I11 = np.array(I11)
    I22 = np.array(I22)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    nsteps = np.linspace(1, len(I00), len(I00))*20000/1E6
    ax.plot(nsteps, I11/I00, color="#7395AE")
    ax.plot(nsteps, I22/I00, color="#5D5C61")
    ax.set_xlabel("Num. MC steps (million)")
    ax.set_ylabel("Covariance ratios")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()