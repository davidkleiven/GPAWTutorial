import h5py as h5
from cemc.phasefield import CahnHilliard, cahn_hilliard_surface_parameter
from cemc.phasefield import GradientCoefficientRhsBuilder
from cemc.phasefield import SlavedTwoPhaseLandauEvaluator
from cemc.phasefield import GradientCoeffNoExplicitProfile
from cemc.tools import TwoPhaseLandauPolynomial
from itertools import product
import numpy as np
from ase.units import kB
import matplotlib as mpl
mpl.rcParams.update({"font.size": 18, "svg.fonttype": "none", "axes.unicode_minus": False})

fname = "data/pseudo_binary_free/adaptive_bias300K_-650mev_bck.h5"
fname = "data/pseudo_binary_free/adaptive_bias300K_-650mev_bck2.h5"
beta = 1.0/(kB*300)


class AlMgSiInterface(GradientCoefficientRhsBuilder):
    def __init__(self, two_phase_poly, end_pts, slope, boundary_values):
        GradientCoefficientRhsBuilder.__init__(self, boundary_values)
        self.poly = two_phase_poly
        self.end_pts = end_pts
        self.slope = slope

    def grad(self, x):
        grad_c = [self.poly.partial_derivative(x[0, i], shape=x[1:, i], var="conc") - self.slope
                  for i in range(x.shape[1])]
        grad1 = [self.poly.partial_derivative(x[0, i], shape=x[1:, i], var="shape", direction=0)
                 for i in range(x.shape[1])]
        grad2 = [self.poly.partial_derivative(x[0, i], shape=x[1:, i], var="shape", direction=1)
                 for i in range(x.shape[1])]
        return np.array([grad_c, grad1, grad2])

    def evaluate(self, x):
        c = x[0, :]
        res = [self.poly.evaluate(x[0, i], shape=x[1:, i]) for i in range(x.shape[1])]
        return np.array(res) - self.slope*(c - self.end_pts[0]) - self.poly.evaluate(self.end_pts[0])

def main():
    with h5.File(fname, 'r') as infile:
        x = np.array(infile["x"])/2000.0
        G = -np.array(infile["bias"])/beta

    N = 4000.0
    vol = 1000.0*4.05**3
    G /= vol
    G *= 1000.0  # Convert to meV
    cahn = CahnHilliard(degree=6, bounds=[0.0, 1.0], penalty=1.0)
    cahn.fit(x, G)
    np.savetxt("data/cahn_coeff.csv", cahn.coeff, delimiter=",")
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, G, drawstyle="steps")
    x_fit = np.linspace(-0.05, 1.05, 100).tolist()

    fit = [cahn.evaluate(x) for x in x_fit]

    ax.plot(x_fit, fit)
    ax.set_xlabel("\$x\$, (\\ce{Al_{2-2x}Mg_xSi_x})")
    ax.set_ylabel("Free energy density (eV/angstrom$^3$)")
    plt.show()


def fit_landau_polynomial():
    from matplotlib import pyplot as plt
    from cemc.phasefield import GradientCoefficient
    with h5.File(fname, 'r') as infile:
        x = np.array(infile["x"])/2000.0
        G = -np.array(infile["bias"])/beta

    N = 4000.0
    vol = 1000.0*4.05**3
    G /= vol
    G -= G[0]
    G *= 1000.0

    poly = TwoPhaseLandauPolynomial(c1=0.0, c2=0.35, conc_order1=3,
                                    conc_order2=3, num_dir=2)
    end1 = int(2*len(G)/5)
    G1 = G[8:end1]
    x1 = x[8:end1]
    start2 = int(3*len(G)/4)
    start2 = end1
    G2 = G[start2:-8]
    x2 = x[start2:-8]
    poly.fit(x1, G1, x2, G2)
    print(poly.coeff_shape)
    x_fit = np.linspace(0.0, 1.0, 600)
    fitted = [poly.evaluate(x_fit[i]) for i in range(len(x_fit))]

    # p, slope = poly.locate_common_tangent(0.0, 1.0)
    # print(p)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, G)
    ax.plot(x_fit, fitted)
    # ax.plot(x_fit, slope*(x_fit-p[0]) + poly.evaluate(p[0]))

    fig2 = plt.figure()
    n_eq = [poly.equil_shape_order(x[i]) for i in range(len(x))]
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(x, n_eq)

    # Explore different shapes
    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1, 1, 1)
    n = np.linspace(0.0, 1.7, 50)
    F = np.zeros((len(n), len(n)))
    #concs = [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    c = 0.9
    for indx in product(range(len(n)), repeat=2):
        value = poly.evaluate(c, shape=np.array([n[indx[0]], n[indx[1]]]))
        F[indx] = value
    ax3.imshow(F, origin="lower", cmap="nipy_spectral")
    plt.show()

    num_density = 0.06
    mJ2ev = 0.0642
    gammas = [2.649180389240096*mJ2ev, 5.901458900832251*mJ2ev]
    grad = poly.slaved_gradient_coefficients(5.0, gammas, num_density)
    print(grad)
    poly.save_poly_terms(fname="data/almgsi_chgl/coeff.csv")


def cahn_hilliard_phase_field():
    from ase.units import J, m
    from phasefield_cxx import PyCahnHilliard
    from phasefield_cxx import PyCahnHilliardPhaseField
    coeff = np.loadtxt("data/cahn_coeff.csv", delimiter=",").tolist()

    surface_tension = 42  # mJ/m^2
    surface_tension *= 1E-3*6.242e+18/1E20  # eV/angstrom^2

    x = np.linspace(0.0, 1.0, 100)
    G = np.polyval(coeff, x)
    dG = G - (x*G[-1] + (1.0-x)*G[0])
    num_per_vol = 0.06  # Al atoms per angstrom

    # Convert to meV (multiply by 1000.0)
    grad_param = 1000.0*cahn_hilliard_surface_parameter(x, dG, surface_tension, num_per_vol)

    cahn_free = PyCahnHilliard(coeff, bounds=[0.0, 1.0], penalty=100.0)
    L = 128  # Angstrom
    M = 1E-3
    dt = 0.01
    alpha = grad_param
    sim = PyCahnHilliardPhaseField(2, L, "data/almgsi_phase_field100/cahn", cahn_free, M, dt, alpha)
    #sim.random_initialization(0.0, 1.0)
    sim.from_file("data/almgsi_phase_field100/cahn00004000000.grid")
    sim.run(3000000, 50000, start=4000000)


def fft_final_state():
    fname = "data/almgsi_phase_field100/cahn00007000000.tsv"
    data = np.zeros((128, 128))
    with open(fname, 'r') as infile:
        for line in infile:
            split = line.split("\t")
            i1 = int(split[0])
            i2 = int(split[1])
            value = float(split[2])
            data[i1, i2] = value
    from matplotlib import pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    nm = data.shape[0]/10.0
    ext = [0, nm, 0, nm]
    ax.imshow(data, cmap="gray", interpolation="gaussian", extent=ext)
    ax.set_xlabel("\$x\$ (nm)")
    ax.set_ylabel("\$y\$ (nm)")

    data -= np.mean(data)
    ft = np.abs(np.fft.fft2(data))
    ft = np.fft.fftshift(ft)
    fig_ft = plt.figure()
    ax_ft = fig_ft.add_subplot(1, 1, 1)
    freq = np.fft.fftshift(np.fft.fftfreq(128, d=0.1))

    start = 30
    stop = 98
    ext = [freq[start], freq[stop], freq[start], freq[stop]]
    ax_ft.imshow(ft[start:stop, start:stop], cmap="gray",
                 interpolation="gaussian", extent=ext)
    ax_ft.set_xlabel("Freqency (nm \$^{-1}\$)")
    ax_ft.set_ylabel("Frequency (nm \$^{-1}\$)")

    fig_rad = plt.figure()
    ax_rad = fig_rad.add_subplot(1, 1, 1)
    rbin, profile = radial_profile(ft)
    freq = rbin/10.0
    ax_rad.plot(freq, profile/np.mean(profile), drawstyle="steps")
    ax_rad.set_xlabel("Frequency (nm\$^{-1})\$")
    ax_rad.set_ylabel("Normalized intensity")
    plt.show()


def radial_profile(data):
    y, x = np.indices((data.shape))
    center = data.shape[0]/2
    r = np.sqrt((x-center)**2 + (y-center)**2)
    r = r.astype(int)
    flat_data = data.ravel()
    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())

    rbin = np.arange(0, np.max(r)+1, 1)
    return rbin, tbin/nr


if __name__ == "__main__":
    # main()
    # cahn_hilliard_phase_field()
    # fft_final_state()
    fit_landau_polynomial()
    
