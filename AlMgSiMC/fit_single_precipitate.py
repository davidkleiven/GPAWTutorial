from scipy.stats import linregress
from apal import SinglePrecipitatePoly
import numpy as np
import h5py as h5
from ase.units import kB
from matplotlib import pyplot as plt
from apal.tools import surface_formation
import json

fname = "data/pseudo_binary_free/adaptive_bias600K_-650mev.h5"
fname_diff = "data/diffraction/layered_bias600K.h5"
outfname = "apal_single_precipitate.json"

beta = 1.0/(kB*600)

def main():
    from matplotlib import pyplot as plt
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

    # Energy is is meV/angstrom

    G_diff = G_diff[2:-2]
    x_diff = x_diff[2:-2]
    rng = np.max(x_diff) - np.min(x_diff)
    x_diff /= rng

    # Symmetrize the function
    G_diff_half = G_diff[int(len(G_diff)/2):]
    G_diff = np.concatenate((G_diff_half[::-1], G_diff_half))

    poly = SinglePrecipitatePoly(c1=0.01, c2=0.96)
    poly.fit(x, G, lim1=(0.0, 0.04), lim2=(0.92, 1.0), show=True)

    gammas = [5.081164748586553, 17.189555738954752, 14.080720554234594]

    # Let's set the thickness of the smallest interface
    # to 10.0 angstrom
    poly.set_landau_barrier_by_interface_thickness(3.0, gammas[0])
    alpha1 = poly.gradient_coefficient(gammas[0])
    alpha2 = poly.gradient_coefficient(gammas[1])

    poly_dict = poly.to_dict()
    poly_dict["gradient_coeff"] = [alpha1, alpha2]
    poly_dict["conc_file"] = fname
    poly_dict["diffraction_file"] = fname_diff

    with open(outfname, 'w') as out:
        json.dump(poly_dict, out)

    print("Coefficients written to {}".format(outfname))

if __name__ == "__main__":
    main()