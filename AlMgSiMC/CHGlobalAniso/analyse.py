import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

N = 1024
folder = "/work/sophus/ChGlobalAniso/TwoAniso/"
def show():
    fname = folder + "cahnHilliard2D_conc_20.bin"
    data = np.fromfile(fname, dtype='>f8')
    data = data.reshape((N, N))
    data -= np.mean(data)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data, cmap="gray")
    plt.show()

def length_scale_evolution():
    prefix = folder + "cahnHilliard2D_conc_"

    lengths = []
    for i in range(1, 100):
        fname = prefix + '{}.bin'.format(i)
        data = np.fromfile(fname, dtype='>f8')
        data = data.reshape((N, N))
        data -= np.mean(data)

        ft = np.abs(np.fft.fft2(data))**2
        freq = np.fft.fftfreq(N)
        FX, FY = np.meshgrid(freq, freq)
        F = np.sqrt(FX**2 + FY**2)
        mean_freq = np.sum(F*ft)/np.sum(ft)
        L = 1.0/mean_freq
        lengths.append(L)
    
    fig = plt.figure()
    t = np.linspace(1.0, len(lengths), len(lengths))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, lengths)
    ax.set_xscale('log')
    ax.set_yscale('log')

    slope, interscept, _, _, _ = linregress(np.log(t)[10:], np.log(lengths)[10:])
    print(interscept, slope)
    ax.plot(t, np.exp(interscept)*t**slope)
    plt.show()


#length_scale_evolution()
show()