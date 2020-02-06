import matplotlib as mpl
mpl.rcParams.update({'axes.unicode_minus': False, 'font.size': 18, 'svg.fonttype': 'none'})
from matplotlib import pyplot as plt
import numpy as np

FOLDER = "/work/sophus/almgsiMgSiRatio"
W = 10
M = 1000
PREFIX_MG = FOLDER + "/ch2species_W{}_M{}_mgconc".format(W, M)
PREFIX_SI = FOLDER + "/ch2species_W{}_M{}_siconc".format(W, M)
N = 128

def get_concs(num):
    fname = PREFIX_MG + "_{}.bin".format(num)
    mg_conc = np.fromfile(fname, dtype='>f8')
    mg_conc = np.reshape(mg_conc, (N, N))

    fname = PREFIX_SI + "_{}.bin".format(num)
    si_conc = np.fromfile(fname, dtype='>f8')
    si_conc = np.reshape(si_conc, (N, N))
    return mg_conc, si_conc

def show_conc(data):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data, cmap="gray")
    return fig, ax


def characteristic_size(indicator):
    indicator -= np.mean(indicator)
    ft = np.abs(np.fft.fft(indicator))**2
    
    avgFreq = 0.0
    for ix in range(N):
        for iy in range(N):
            fx = ix/N
            fy = iy/N
            if ix > N/2:
                fx -= 1.0
            if iy > N/2:
                fy -= 1.0
            fr = np.sqrt(fx**2 + fy**2)
            avgFreq += fr*ft[ix, iy]
    avgFreq /= ft.sum()
    return 1.0/avgFreq
            


def main():
    global PREFIX_MG, PREFIX_SI
    W_values = [10, 50, 100, 500]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    colors = ['#39362e', '#a6463c', '#ba9666', '#615c54']
    for j, w in enumerate(W_values):
        mg_concs = []
        si_concs = []
        for i in range(500):
            mg, si = get_concs(i)
            PREFIX_MG = FOLDER + "/ch2species_W{}_M{}_mgconc".format(w, M)
            PREFIX_SI = FOLDER + "/ch2species_W{}_M{}_siconc".format(w, M)
            tot = mg + si
            mask = tot > 0.45
            mean_mg = mg[mask].mean()
            mean_si = si[mask].mean()
            mg_concs.append(mean_mg)
            si_concs.append(mean_si)
        ax.plot(mg_concs, color=colors[j], marker='v', mfc='none', markersize=5, markevery=25)
        ax.plot(si_concs, color=colors[j], marker='o', mfc='none', markersize=5, markevery=25)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Time (-)")
    ax.set_ylabel("Concentration")
    plt.show()


main()
