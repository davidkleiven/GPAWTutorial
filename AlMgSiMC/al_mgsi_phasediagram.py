from cemc.tools import BinaryCriticalPoints 
import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'axes.unicode_minus': False, 'font.size': 18, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
import h5py as h5
from scipy.stats import linregress
from mpltools import color
from cemc.phasefield.tools import cahn_hilliard_surface_parameter
import json
from cemc.phasefield.phasefield_util import fit_kernel
from phasefield_cxx import PyGaussianKernel



fit_start_stop = {
    400: [[0, 10], [-40, -10]],
    500: [[0, 10], [-40, -10]],
    600: [[0, 20], [-60, -1]],
    700: [[0, 80], [-100, -30]],
    800: [[60, 100], [-200, -100]]
}


def construct_cahn_hilliard_model(fname, T):
    from ase.units import kB
    with h5.File(fname, 'r') as infile:
        x = np.array(infile["x"])/2000.0
        G = -np.array(infile["bias"])*kB*T

    res = linregress(x, G)
    slope = res[0]
    interscept = res[1]
    G -= (slope*x + interscept)
    N = 4000.0
    vol = 1000.0*4.05**3
    G /= vol
    G *= 1000.0

    x1_approx_indx = np.argmin(G[x < 0.4])
    x2_approx_indx = np.argmin(G[x>0.6]) + len(G[x<0.6])
    print(x[x1_approx_indx], x[x2_approx_indx])

    slope = (G[x2_approx_indx] - G[x1_approx_indx])/(x[x2_approx_indx] - x[x1_approx_indx])
    
    G -= (slope*(x - x[x1_approx_indx]) + G[x1_approx_indx])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, G)
    x_fit = np.linspace(-0.1, 1.1, 100)

    x_pad = np.linspace(-0.2, 0.0, 20)
    G_pad = -x_pad + G[0]

    G_fit = np.concatenate((G_pad, G))
    x_fit = np.concatenate((x_pad, x))
    print(len(G_fit), len(x_fit))
    poly = np.polyfit(x_fit, G_fit, 20)
    ax.plot(x_fit, np.polyval(poly, x_fit))
    ax.plot(x_fit, G_fit)
    plt.show()

    # Save data
    data = {}
    data['poly'] = poly.tolist()

    interf_energy = (5.081164748586553 + 17.189555738954752)*0.5
    data['alpha'] = cahn_hilliard_surface_parameter(x[x1_approx_indx:x2_approx_indx], G[x1_approx_indx:x2_approx_indx], interf_energy)
    fname_out = fname.split('.')[0] + "_cahn_hilliard.json"
    with open(fname_out, 'w') as outfile:
        json.dump(data, outfile)
    print("Cahn-Hilliard data written to {}".format(fname_out))





def plot_all_free_energies():
    temps = [500, 600, 700, 800]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    colors = ['']
    map_color = color.color_mapper([500, 800], cmap='copper')
    for T in temps:
        fname = "data/pseudo_binary_free/adaptive_bias{}K_-650mev.h5".format(T)
        with h5.File(fname, 'r') as infile:
            x = np.array(infile["x"])/2000.0
            G = -np.array(infile["bias"])

        res = linregress(x, G)
        slope = res[0]
        interscept = res[1]
        G -= (slope*x + interscept)
        G -= G[0]
        ax.plot(x, G, drawstyle='steps', color=map_color(T))
    ax.set_ylabel("\$\\beta \Delta G\$")
    ax.set_xlabel("MgSi concentration")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

def critical(fname, temp):
    with h5.File(fname, 'r') as infile:
        x = np.array(infile["x"])/2000.0
        G = -np.array(infile["bias"])

    res = linregress(x, G)
    slope = res[0]
    interscept = res[1]
    G -= (slope*x + interscept)

    bc = BinaryCriticalPoints()

    x1_approx_indx = np.argmin(G[x < 0.4])
    x2_approx_indx = np.argmin(G[x>0.6])


    st1 = fit_start_stop[temp][0][0]
    end1 = fit_start_stop[temp][0][1]
    st2 = fit_start_stop[temp][1][0]
    end2 = fit_start_stop[temp][1][1]

    p1 = np.polyfit(x[st1:end1], G[st1:end1], 2)
    p2 = np.polyfit(x[st2:end2], G[st2:end2], 2)
    x1, x2 = bc.coexistence_points(p1, p2)

    p_spin = np.polyfit(x, G, 10)
    
    spin = bc.spinodal(p_spin)
    spin1 = spin[spin > x1]
    x1_spin = np.min(spin1)
    spin2 = spin[spin < x2]
    x2_spin = np.max(spin2)

    print(x1, x2, x1_spin, x2_spin)
    bc.plot(x, G, polys=[p1, p2, p_spin])
    plt.show()


def fit_ideal_expression(T, x):
    y = np.log(x/(1-x))
    slope, interscept, _, _, _ = linregress(1/T, y)
    # plt.plot(1/T, y)
    # plt.plot(1/T, interscept + slope/T)
    #plt.show()
    return interscept, slope


def ideal_func(interscept, slope, T):
    exponent = interscept + slope/T
    return np.exp(exponent)/(1 + np.exp(exponent))


def plot_phasediagram():
    fname = "data/al_mgsi_phasediagram.csv"
    data = np.loadtxt(fname, delimiter=',')
    T = data[:, 0]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(data[:, 1], T, "o", mfc="none", color="#557A95")
    T_fit = np.linspace(500, 900, 101)
    interscept, slope = fit_ideal_expression(T, data[:, 1])
    ideal = ideal_func(interscept, slope, T_fit)
    ax.plot(ideal, T_fit, color="#557A95")
    ax.fill_between(ideal, T_fit, 850, color="#B1A296")

    ax.plot(data[:, 2], T, "o", mfc="none", color="#557A95")
    interscept, slope = fit_ideal_expression(T, data[:, 2])
    ideal = ideal_func(interscept, slope, T_fit)
    ax.plot(ideal, T_fit, color="#557A95")
    ax.fill_between(ideal, T_fit, 850, color="#B1A296")

    ax.plot(data[:, 3], T, ls='--', marker="o", color="#5D5C61")
    #ax.plot(np.polyval(poly, T_fit), T_fit)
    ax.plot(data[:, 4], T, ls='--', marker="o", color="#5D5C61")
    ax.set_ylim(500, 800)
    ax.set_xlim(0, 1)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("MgSi concentration")
    ax.set_ylabel("Temperature (K)")
    plt.show()

def main():
    #critical("data/pseudo_binary_free/adaptive_bias400K_-650mev.h5", 400)
    critical("data/pseudo_binary_free/adaptive_bias500K_-650mev.h5", 500)
    critical("data/pseudo_binary_free/adaptive_bias600K_-650mev.h5", 600)
    critical("data/pseudo_binary_free/adaptive_bias700K_-650mev.h5", 700)
    critical("data/pseudo_binary_free/adaptive_bias800K_-650mev.h5", 800)

if __name__ == "__main__":
    #main()
    #plot_phasediagram()
    #plot_all_free_energies()
    construct_cahn_hilliard_model("data/pseudo_binary_free/adaptive_bias500K_-650mev.h5", 500)
