import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 18, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
from matplotlib import pyplot as plt
from scipy.stats import linregress
from matplotlib import cm
from copy import deepcopy


def fft(fname, show=False):
    data = np.zeros((1024, 1024))
    with open(fname, 'r') as infile:
        for line in infile:
            split = line.split("\t")
            i1 = int(split[0])
            i2 = int(split[1])
            value = float(split[2])
            data[i1, i2] = value

    from matplotlib import pyplot as plt

    if show:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        nm = data.shape[0]/10.0
        ext = [0, nm, 0, nm]
        ax.imshow(data, cmap="gray", interpolation="gaussian", extent=ext)
        ax.set_xlabel("\$x\$ (nm)")
        ax.set_ylabel("\$y\$ (nm)")

    data -= np.mean(data)
    freq = np.fft.fftshift(np.fft.fftfreq(1024, d=0.1))

    ft = np.abs(np.fft.fft2(data))
    ft = np.fft.fftshift(ft)
    if show:
        fig_ft = plt.figure()
        ax_ft = fig_ft.add_subplot(1, 1, 1)
    

        start = 30
        stop = 98
        ext = [freq[start], freq[stop], freq[start], freq[stop]]
        ax_ft.imshow(ft, cmap="gray",
                    interpolation="gaussian", extent=ext)
        ax_ft.set_xlabel("Freqency (nm \$^{-1}\$)")
        ax_ft.set_ylabel("Frequency (nm \$^{-1}\$)")

    rbin, profile = radial_profile(ft)
    freq = 10*rbin/data.shape[0]

    if show:
        fig_rad = plt.figure()
        ax_rad = fig_rad.add_subplot(1, 1, 1)
        
        ax_rad.plot(freq, profile/np.mean(profile), drawstyle="steps")
        ax_rad.set_xlabel("Frequency (nm\$^{-1})\$")
        ax_rad.set_ylabel("Normalized intensity")
        plt.show()
    return freq, profile


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


def calculate_average_lengthscales(prefix):
    folder = "data/almgsi_ch500K_run_truncnorm"
    folder = "/work/sophus/almgsi_ch500K_run2"
    time_data = np.loadtxt(folder + "/" + prefix + "__adaptive_time.csv", delimiter=",")

    mean_freq = []
    std_freq = []
    time = []
    for n in range(1000, 50001, 1000):
        print("Calculating", n)
        if n < 10000:
            suffix = "_0000000{}.tsv".format(n)
        else:
            suffix = "_000000{}.tsv".format(n)

        fname = folder + "/" + prefix + suffix
        try:
            freq, profile = fft(fname)
        except Exception as exc:
            print(exc)
            continue
        if np.mean(profile) < 1E-6:
            continue
        avg = np.sum(freq*profile)/np.sum(profile)
        variance = np.sum((freq-avg)**2 * profile)/np.sum(profile)
        mean_freq.append(avg)
        std_freq.append(np.sqrt(variance))

        closest_time_indx = np.argmin(np.abs(time_data[:, 0] - n))
        time.append(time_data[closest_time_indx, 1])

    res = np.vstack((time, mean_freq, std_freq)).T
    outfname = folder + "/postproc/{}_frequency_statistics.csv".format(prefix)
    np.savetxt(outfname, res, delimiter=",")
    print("Frequency statistics written to {}".format(outfname))


def plot_frequency_statistics():
    folder = "data/almgsi_ch500K_run_truncnorm/postproc"
    folder = "/work/sophus/almgsi_ch500K_run2/postproc"
    fnames = ["chgl_50_frequency_statistics.csv", "chgl_40_frequency_statistics.csv",
              "chgl_30_frequency_statistics.csv", "chgl_20_frequency_statistics.csv",
              "chgl_10_frequency_statistics.csv"]

    fnames = [folder + "/" + f for f in fnames]

    concs = [0.5, 0.4, 0.3, 0.2, 0.1]
    orig_concs = deepcopy(concs)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    freqs = []
    profiles = []
    markers = ['o', 'd', 'v', '^', 'h', 'p']

    concs = np.array(concs)
    concs -= np.min(concs)
    concs /= np.max(concs)
    col = 1

    if col == 2:
        print("Plotting second moment of distribution")
    elif col == 1:
        print("Plotting first moment of distribution")
    else:
        raise ValueError("col has to be 1 or 2")

    for i, fname in enumerate(fnames):
        data = np.loadtxt(fname, delimiter=",")
        freqs.append(data[:, 0])
        profiles.append(data[:, col])
        ax.plot(data[:, 0], data[:, col], markers[i], mfc="none",
                color=cm.copper(concs[i]), ms=8, markeredgewidth=1.5,
                label="{}%".format(int(100*orig_concs[i])))

    # Fit a linear line to beginning
    slope, interscept, _, _, _ = linregress(
        np.log(freqs[0][:12]), np.log(profiles[0][:12]))

    print("Slope (exponent)", slope)
    freq_fit = np.logspace(-1.5, 3, 10)
    ax.plot(freq_fit, np.exp(interscept)*freq_fit**slope, ls='--', color="#5d5C61")
    ax.set_xscale("log")
    ax.set_yscale("log", basey=2)
    ax.set_xlabel("Dimensionless time")
    ax.set_ylabel("Frequency (nm\$^{-1}\$)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend(frameon=False)
    plt.show()

def plot_profiles():
    fnames = ["chgl_10_00000001000.tsv", "chgl_10_00000005000.tsv", "chgl_10_00000010000.tsv",
              "chgl_10_00000030000.tsv", "chgl_10_00000040000.tsv", "chgl_10_00000050000.tsv"]
    times = list(range(len(fnames)))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    folder = "data/almgsi_ch500K_run3"
    folder = "data/almgsi_ch500K_run_truncnorm"

    times = np.array(times, dtype=np.float64)
    times -= np.min(times)
    times /= np.max(times)
    times = times.tolist()
    
    for t, fname in zip(times, fnames):
        full_fname = folder + "/" + fname
        freq, prof = fft(full_fname)
        print(np.mean(prof))
        ax.plot(freq, prof/np.sum(prof), drawstyle='steps', color=cm.copper(t))
    ax.set_yscale('log')
    ax.set_xlabel("Frequency (nm \${-1}\$)")
    ax.set_ylabel("Occurence probability")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()


def main():
    #fft('data/almgsi_ch500K_run2/chgl_50_00000001000.tsv')
    #plot_profiles()
    #calculate_average_lengthscales("chgl_50")
    plot_frequency_statistics()

main()