import h5py as h5
from cemc.phasefield import CahnHilliard, cahn_hilliard_surface_parameter
from cemc.phasefield import GradientCoefficientRhsBuilder
from cemc.phasefield import SlavedTwoPhaseLandauEvaluator
from cemc.phasefield import GradientCoeffNoExplicitProfile
from cemc.phasefield import TwoPhaseLandauPolynomial
from itertools import product
import numpy as np
from ase.units import kB
import matplotlib as mpl
from scipy.signal import decimate
from cemc.phasefield import PeakPosition
from cemc.phasefield import StraightLineSaddle
from cemc.phasefield import InteriorMinima
import json
from scipy.stats import linregress
mpl.rcParams.update({"font.size": 18, "svg.fonttype": "none", "axes.unicode_minus": False})

fname = "data/pseudo_binary_free/adaptive_bias300K_-650mev_bck.h5"
fname = "data/pseudo_binary_free/adaptive_bias300K_-650mev_bck2.h5"
fname = "data/pseudo_binary_free/adaptive_bias600K_-650mev.h5"
fname_diff = "data/diffraction/layered_bias600K.h5"
beta = 1.0/(kB*400)


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

    with h5.File(fname_diff, 'r') as infile:
        x_diff = np.array(infile["x"])
        G_diff = -np.array(infile["bias"])/beta

    #G_diff = decimate(G_diff, 4)
    #x_diff = x_diff[2::4]
    x_diff_range = x_diff[-1] - x_diff[0]
    N = 4000.0
    vol = 1000.0*4.05**3
    G /= vol
    G_diff /= vol
    G -= G[0]
    G_diff -= G_diff[0]
    G *= 1000.0
    G_diff *= 1000.0
    res = linregress(x, G)
    slope = res[0]
    interscept = res[1]
    G -= (slope*x + interscept)

    G_diff = G_diff[2:-2]
    x_diff = x_diff[2:-2]

    # Symmetrize the function
    G_diff_half = G_diff[int(len(G_diff)/2):]
    G_diff = np.concatenate((G_diff_half[::-1], G_diff_half))

    # Align the left end
    params_landau = {
        "c1": 0.0,
        "c2": 0.96,
        "conc_order1": 16,
        "conc_order2": 1,
    }
    poly = TwoPhaseLandauPolynomial(**params_landau)
    weights = {"eq_phase2": 0.0, "eq_phase1": 1.0}
    poly.fit(x, G, weights=weights, kernel_width=0.01, num_kernels=200, width=None, smear_width=0, shape="auto", lamb=1E-1)
    print(poly.coeff_shape)
    n_eq = poly.equil_shape_order(x)
    n_eq_max = np.max(n_eq)
    fitted_end_point = poly.evaluate(1.0)
    G_diff -= G_diff[-1]
    x_diff *= (n_eq_max/x_diff_range)

    shape_param = np.zeros((len(x_diff), 3))
    shape_param[:, 0] = x_diff
    shape_param[:, 1] = np.max(x_diff) - x_diff

    peak_const = PeakPosition(weight=100.0, peak_indx=int(len(G_diff)/2),
                              conc=0.95, eta=shape_param, free_eng=G_diff)
    saddle = StraightLineSaddle(weight=5E-5, eta=shape_param, conc=0.95,
                                normal=[1.0, 1.0, 0.0])

    interior = InteriorMinima(weight=1000.0, conc=0.95, eta_min=0.1, eta_max=0.9, num_eta=50)
    poly.fit_fixed_conc_varying_eta(0.95, shape_param, G_diff,
                                    weights={"peak": 0.0,
                                             "mixed_peaks": 100000000.0},
                                    constraints=[peak_const, saddle, interior])
    poly.save_poly_terms(fname="data/polyterms_fit.json")
    # Fit off axis
    x_fit = np.linspace(0.0, 1.0, 600)
    fitted = poly.evaluate(x_fit)
    poly.plot_individual_polys()

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, G)
    ax.plot(x_fit, fitted)
    ax.set_xlabel("Fraction of MgSi")
    ax.set_ylabel("Free energy density (meV/angstrom$^3$)")

    fig2 = plt.figure()

    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.plot(x, n_eq)
    ax2.set_ylabel("Degree of layering")
    ax2.set_xlabel("MgSi concentration")

    # Explore different shapes
    fitted_diff = poly.evaluate(0.95, shape=shape_param)
    n = np.linspace(0.0, 1.1, 50)
    F = np.zeros((len(n), len(n)))
    c = 0.6
    for c in [0.1, 0.3, 0.5, 0.7, 0.9]:
        fig3 = plt.figure()
        ax3 = fig3.add_subplot(1, 1, 1)
        for indx in product(range(len(n)), repeat=2):
            value = poly.evaluate(c, shape=[n[indx[0]], n[indx[1]], 0.0])
            F[indx] = value
        extent = [0.0, 1.1, 0.0, 1.1]
        im = ax3.imshow(F, origin="lower", cmap="nipy_spectral", extent=extent,
                        vmin=np.min(F), vmax=np.max(G_diff))
        levels = np.linspace(np.min(F), np.max(fitted_diff), 20)
        ax3.contour(F, levels=levels, origin="lower", colors="white", extent=extent)
        ax3.set_xlabel("Degree of layering along $x$")
        ax3.set_ylabel("Degree of layering along $y$")
        cbar = fig3.colorbar(im)
        cbar.set_label("Free energy density (meV/angstrom$^3$)")
        plt.show()
        fig3.savefig("data/free_energy_landscape{}.svg".format(int(100*(c))))
        fig3.savefig("data/free_energy_landscape{}.png".format(int(100*c)))
        plt.close(fig3)


    # Plot the diffraction barrier
    fig4 = plt.figure()
    ax_diff = fig4.add_subplot(1, 1, 1)
    ax_diff.plot(x_diff, G_diff, drawstyle="steps")

    
    ax_diff.plot(x_diff, fitted_diff)
    plt.show()

    #num_density = 0.06
    num_density = 1.0
    mJ2ev = 0.0642
    gammas = [2.649180389240096*mJ2ev, 5.901458900832251*mJ2ev, 
              7.169435749573065*mJ2ev]
    gammas = [5.081164748586553, 17.189555738954752, 14.080720554234594]

    # Find the gradient coefficients
    evaluator = SlavedTwoPhaseLandauEvaluator(poly, involution_order=30)
    nmax = np.sqrt(2.0/3.0)
    boundary = {
        (0, 1): [[0.0, 1.0], [0.0, nmax], [0.0, 0.0]],
        (0, 2): [[0.0, 1.0], [0.0, 0.0], [0.0, nmax]],
        (1, 2): [[1.0, 1.0], [1E-6, nmax], [nmax, 1E-6]]
    }

    interface_energy = {
        (0, 1): gammas[0],
        (0, 2): gammas[1],
        (1, 2): gammas[2]
    }
    params_vary = {
        (0, 1): [0, 1],
        (0, 2): [0, 2],
        (1, 2): [1, 2]
    }
    grad_coeff_finder = GradientCoeffNoExplicitProfile(
        evaluator, boundary, interface_energy,
        params_vary, num_density, apply_energy_correction=True,
        init_grad_coeff=[5.0, 5.0, 5.0], neg2zero=True)
    grad_coeff = grad_coeff_finder.solve()

    alpha = 2.0 

    poly_dict = poly.to_dict()
    poly_dict["gradient_coeff"] = grad_coeff.tolist()
    poly_dict["conc_file"] = fname
    poly_dict["diffraction_file"] = fname_diff
    poly_dict["landau_params"] = params_landau

    json_fname = "chgl_almgsi_linfit.json"
    with open(json_fname, 'w') as outfile:
        json.dump(poly_dict, outfile, indent=2)
    print("JSON file: {}".format(json_fname))

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
    #num_per_vol = 1.0

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
    
