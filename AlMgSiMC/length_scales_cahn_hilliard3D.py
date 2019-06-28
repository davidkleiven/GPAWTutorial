import matplotlib as mpl
mpl.rcParams.update({'axes.unicode_minus': False, 'font.size': 18, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy.stats import linregress
from scipy.optimize import curve_fit

PREFIX = "data/almgsi_ch500K_3D/chgl_"
PREFIX_POST = "data/almgsi_ch500K_3D/postproc/"


def fit_func(x, A, B, a, b):
    return A*x**a/(1 + B**2*x**b)


def exp_decay(x, A, B, asymp, slope, exponent):
    eff_exp = asymp + slope*np.exp(-exponent**2 * x)
    return A + B*x**eff_exp


def fit_cross_over(x, y):
    """
    Fit a function of the form Ax^a/(1 + Bx* b)
    """
    
    x0 = [1, 1, 0.0, 0.33]
    popt, pcov = curve_fit(fit_func, x, y, p0=x0)
    return popt


def fit_exp_decay(x, y):
    """
    Fit a function of the form Ax^a/(1 + Bx* b)
    """
    x0 = [1.0, 1.0, -0.33, 0.0, 0.0]
    popt, pcov = curve_fit(exp_decay, x, y, p0=x0)
    return popt


def calculate_radial_average(fname):
    data = np.loadtxt(fname, delimiter="\t")
    data = np.zeros((128, 128, 128))
    radius = np.zeros((128, 128, 128), dtype=np.int32)
    c = 64
    with open(fname, 'r') as infile:
        for line in infile:
            split = line.split("\t")
            i1 = int(split[0])
            i2 = int(split[1])
            i3 = int(split[2])
            value = float(split[3])
            data[i1, i2, i3] = value
            radius[i1, i2, i3] = int(np.sqrt((i1-c)**2 + (i2-c)**2 + (i3-c)**2))

    outfile = PREFIX_POST + fname.rsplit("/")[-1]
    outfile = outfile.split(".")[0] + ".csv"

    data -= np.mean(data)

    ft = np.abs(np.fft.fftn(data))
    ft = np.fft.fftshift(ft)

    histogram = np.bincount(radius.ravel(), weights=ft.ravel())
    nr = np.bincount(radius.ravel())
    rbin = np.arange(0, np.max(radius)+1, 1)
    np.savetxt(outfile, np.vstack((rbin, histogram, nr)).T, header="Rad. pixel, Sum, Num. occurences")
    print("Data written to {}".format(outfile))

def plot_profiles_vs_time():
    fnames = ["chgl_10_00000001000.csv", "chgl_10_00000010000.csv", "chgl_10_00000100000.csv", "chgl_10_00000550000.csv"]
    times = [15.54, 113.375, 769.77, 3899]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ["#5d5c61", "#379683", "#557a95", "#7395ae", "#b1a296"]
    for i, fname in enumerate(fnames):
        r, hist, num_occ = np.loadtxt(PREFIX_POST + fname, unpack=True)
        y = hist[1:]/num_occ[1:]
        y /= np.sum(y)
        ax.plot(r[1:], y, drawstyle="steps", color=colors[i], label="{}".format(times[i]))
        #ax.fill_between(r[1:], 1.0, y, color=colors[i], alpha=0.2)
    ax.set_yscale("log")
    ax.set_xlabel("Spatial frequency (nm\$^{-1}\$)")
    ax.set_ylabel("Relative occurence")
    ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

def extract_profiles_conc(conc):
    for iteration in range(1000, 900000, 1000):
        fname = PREFIX + str(conc) + "_00000"
        print("Current iteration {}".format(iteration))
        if iteration < 100000:
            fname += "0"
        if iteration < 10000:
            fname += "0"

        fname += "{}.tsv".format(iteration)

        if os.path.exists(fname):
            calculate_radial_average(fname)
        else:
            print("Skipping file {}".format(fname))


def extract_profiles():
    for conc in [10, 20, 30, 40, 50]:
        print("Current concentration {}".format(conc))
        extract_profiles_conc(conc)


def plot_scale_distribution():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    fig_exp = plt.figure()
    ax_exp = fig_exp.add_subplot(1, 1, 1)

    colors = {
        10: "#5d5c61",
        20: "#379683",
        30: "#557a95",
        40: "#7395ae",
        50: "#b1a296"
    }

    for conc in [10, 20, 30, 40, 50]:
        time_file = PREFIX + "{}__adaptive_time.csv".format(conc)
        iteration, time = np.loadtxt(time_file, delimiter=",", unpack=True)

        avg_freqs = []
        for it in iteration.tolist():
            fname = PREFIX_POST + "chgl_{}_00000".format(conc)

            if it < 100000:
                fname += "0"
            if it < 10000:
                fname += "0"
            fname += "{}.csv".format(int(it))

            try:
                radius, histogram, num_occ = np.loadtxt(fname, unpack=True)
            except OSError:
                break
            histogram /= num_occ
            histogram /= np.sum(histogram)
            avg_freq = np.sum(radius*histogram)
            avg_freqs.append(avg_freq)

        # Convert
        dx = 1.0 # nm
        radius *= 1.0/dx
        ax.plot(time[:len(avg_freqs)], avg_freqs, "o", mfc="none",  color=colors[conc], markersize=5, label="{}".format(conc))
        start = int(len(avg_freqs)/8)
        end = len(avg_freqs)
        result = linregress(np.log(time[start:end]), np.log(avg_freqs[start:]))
        slope = result[0]
        interscept = result[1]
        print("Slope for conc.: {}={}".format(conc, slope))
        t_fit = np.linspace(1, 4E4, 100)
        fitted = np.exp(interscept)*t_fit**slope
        if conc == 40:
            ax.plot(t_fit, fitted, "--", color="grey")#, )


        # poly = np.polyfit(np.log(time[start:end]), np.log(avg_freqs[start:]), deg=2)
        # exponent = np.polyder(poly)
        # ax_exp.plot(t_fit, np.polyval(exponent, np.log(t_fit)), color=colors[conc])

        # coeff = fit_cross_over(time[:end], avg_freqs[start:])
        # coeff_exp = fit_exp_decay(time[start:end], avg_freqs[start:])
        # print(coeff_exp)
        # ax.plot(t_fit, fit_func(t_fit, coeff[0], coeff[1], coeff[2], coeff[3]), "-.", color=colors[conc])
        # print("Conc", coeff)
        # ax.plot(t_fit, np.exp(np.polyval(poly, np.log(t_fit))), "-.", color=colors[conc])
    ax.legend(frameon=False)
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel("Average spatial frequency")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xscale("log")
    ax.set_yscale("log", basey=2)
    xtick = ax.get_xticks()
    latex_ticks = [r"\$10^{{{}}}\$".format(int(np.log10(x))) for x in xtick]
    ax.set_xticklabels(latex_ticks)
    yticks = ax.get_yticks()
    latex_yticks = [r"\$2^{{{}}}\$".format(int(np.log2(x))) for x in yticks]
    ax.set_yticklabels(latex_yticks)

    # Exponent plot
    ax_exp.set_xscale("log")
    plt.show()


if __name__ == "__main__":
    #extract_profiles()
    #plot_scale_distribution()
    plot_profiles_vs_time()