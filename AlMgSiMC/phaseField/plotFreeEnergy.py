from matplotlib import pyplot as plt
import numpy as np

W = 1.0
def free_energy(c1, c2):
    s = c1 + c2
    d = c1 - c2
    return 16.0*(s**2 - 2*s**3 + s**4) + W*d**2

def plotFree():
    c1 = np.linspace(0.0, 1.0, 100)
    c2 = np.linspace(0.0, 1.0, 100)
    C1, C2 = np.meshgrid(c1, c2)
    F = free_energy(C1, C2)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(F, cmap="terrain", vmin=0.0, vmax=4.0, origin="lower", extent=[0.0, 1.0, 0.0, 1.0])
    ax.contour(F, colors="grey", levels=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4],  extent=[0.0, 1.0, 0.0, 1.0])
    cbar = fig.colorbar(im)
    cbar.set_label("Free energy")
    ax.set_xlabel("Mg concentration")
    ax.set_ylabel("Si concentration")
    plt.show()

plotFree()