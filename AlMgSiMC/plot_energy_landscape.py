import matplotlib as mpl
mpl.rcParams.update({'font.size': 18, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
from matplotlib import pyplot as plt
import numpy as np


# def func700K(c, n1, n2):
#     return 2.08*c**2 - 0.04*c - 0.1 - 5.92*c*(n1**2 + n2**2) + 5.92*(n1**2 + n2**2) -\
#             12.4*(n1**4 + n2**4 - n1**2*n2**2 - n1**6 - n2**6) - \
#             4.15*(n1**2*n2**4 + n2**2*n1**4)

def func700K(c, n1, n2):
    return 1.57*c**2 - 0.09*c - 0.08 - 4.16*c*(n1**2 + n2**2) + 3.77*(n1**2 + n2**2) -\
            8.29*(n1**4 + n2**4 - n1**2*n2**2 - n1**6 - n2**6) - \
            2.76*(n1**2*n2**4 + n2**2*n1**4)


def plot_conc_eta():
    c = np.linspace(0.0, 1.0, 100)
    eta = np.linspace(0.0, 1.0, 100)
    C, N = np.meshgrid(c, eta)
    z = func700K(C, N, 0.0)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    extent = [C.min(), C.max(), N.min(), N.max()]
    im = ax.imshow(z, extent=extent, cmap='terrain', origin='lower', vmax=0.5)
    ax.set_xlabel('MgSi concentration')
    ax.set_ylabel('Order parameter')
    levels = np.linspace(z.min(), 0.5, 20)
    ax.contour(C, N, z, levels=levels, colors='grey')
    cb = fig.colorbar(im)
    cb.set_label('Free energy (meV/Å\$^3\$)')
    plt.show()


if __name__ == '__main__':
    plot_conc_eta()
