import h5py as h5
import numpy as np
from scipy.stats import linregress
import matplotlib as mpl
import json
mpl.rcParams.update({"axes.unicode_minus": False, "font.size": 18, "svg.fonttype": "none"})


def free_energy_vs_comp():
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    temps = [400, 500, 600, 650, 700, 750, 800]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cNorm = Normalize(vmin=400, vmax=800)
    scalarMap = ScalarMappable(norm=cNorm, cmap="copper")
    for T in temps:
        fname = "data/pseudo_binary_free/adaptive_bias{}K_-650mev.h5".format(T)
        with h5.File(fname, 'r') as hfile:
            x = np.array(hfile["x"])/2000.0
            betaG = -np.array(np.array(hfile["bias"]))
        value = (T - 400.0)/400.0
        color = scalarMap.to_rgba(T)
        res = linregress(x, betaG)
        slope = res[0]
        interscept = res[1]
        betaG -= (slope*x + interscept)
        betaG -= betaG[0]
        ax.plot(x, betaG, drawstyle="steps", color=color)
    ax.set_xlabel("Fraction MgSi")
    ax.set_ylabel("\$\\beta \Delta G\$")
    scalarMap.set_array([400.0, 800])
    cbar = fig.colorbar(scalarMap, orientation="horizontal", fraction=0.07, anchor=(1.0, 0.0))
    cbar.set_label("Temperature (K)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    plt.show()


def free_energy_chem_pot():
    from matplotlib import pyplot as plt
    from scipy.signal import savgol_filter
    fname = "data/pseudo_binary_free/adaptive_bias600K_-650mev.h5"
    with h5.File(fname, 'r') as hfile:
        x = np.array(hfile["x"])/2000.0
        betaG = -np.array(np.array(hfile["bias"]))

    res = linregress(x, betaG)
    slope = res[0]
    interscept = res[1]
    betaG -= (slope*x + interscept)
    coeff = np.polyfit(x, betaG, 12)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, betaG, drawstyle="steps", color="#5D5C61")
    # ax.plot(x, np.polyval(coeff, x))

    mu_coeff = np.polyder(coeff, 1)
    mu = np.polyval(mu_coeff, x)
    ax2 = ax.twinx()
    ax2.plot(x, mu/1000.0, ls="--", color="#557A95")
    ax2.set_ylim([-1.5, 2.0])
    ax.set_ylabel("\$\Delta G\$")
    ax2.set_ylabel("\$\\beta \mu \\times 10^\{-3\}\$")
    ax.set_xlabel("Fraction MgSi")
    plt.show()

    

def reflection():
    from matplotlib import pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    temps = [600, 700, 800]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cNorm = Normalize(vmin=600, vmax=800)
    scalarMap = ScalarMappable(norm=cNorm, cmap="copper")
    for T in temps:
        fname = "data/diffraction/layered_bias{}K.h5".format(T)
        with h5.File(fname, 'r') as hfile:
            x = np.array(hfile["x"])
            betaG = -np.array(np.array(hfile["bias"]))
        color = scalarMap.to_rgba(T)
        res = linregress(x, betaG)
        slope = res[0]
        interscept = res[1]
        betaG -= np.min(betaG)
        ax.plot(x, betaG, drawstyle="steps", color=color, lw=2)
    ax.set_xlim([0.25, 0.5])
    ax.set_xlabel("Normalised diffraction intensity")
    ax.set_ylabel("\$\\beta \Delta G\$")
    scalarMap.set_array([500.0, 800])
    #cbar = fig.colorbar(scalarMap, orientation="horizontal", fraction=0.07, anchor=(1.0, 0.0))
    #cbar.set_label("Temperature (K)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    data = load_fit_data()
    #ax.plot(data["MCpoints"]["x"], data["MCpoints"]["y"])
    ax.plot(data["Fitted"]["x"], data["Fitted"]["y"], ls="--", color="black")
    plt.show()

def load_fit_data():
    from ase.units import kB
    fname = "data/free_energy.json"
    with open(fname, 'r') as infile:
        data = json.load(infile)
    
    data_fit = {}
    T = 600
    for item in data["datasetColl"]:
        xy = {"x": [], "y": []}
        for subitem in item["data"]:
            xy["x"].append(subitem["value"][0]/(2*np.sqrt(2.0/3.0)))
            xy["y"].append(4.05**3*subitem["value"][1]/(kB*T))
        data_fit[item["name"]] = xy
    return data_fit

def solubility():
    from matplotlib import pyplot as plt
    data = np.loadtxt("data/location_of_minima.csv", delimiter=",")
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data[:, 1], data[:, 0], marker="o", mfc="none", color="#557A95")
    ax.fill_between(data[:, 1], data[:, 0], 900, color="#B1A296")
    ax.set_ylim([400, 900])
    ax.set_xlim([0.0, 0.14])
    ax.set_xlabel("Fraction MgSi")
    ax.set_ylabel("Temperature  (K)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    #free_energy_vs_comp()
    #solubility()
    reflection()
    #free_energy_chem_pot()
