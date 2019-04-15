import sys
import numpy as np
from matplotlib import pyplot as plt

USAGE_MSG = "Usage: python chgl_noise_correlation.py <tab_separated_file>"


def main(fname):
    x_indx = []
    y_indx = []
    value = []
    with open(fname, 'r') as infile:
        for i, line in enumerate(infile):
            split = line.rstrip().split("\t")
            x_indx.append(int(split[0]))
            y_indx.append(int(split[1]))
            value.append(float(split[2]))
    
    Lx = np.max(x_indx) + 1
    Ly = np.max(y_indx) + 1

    values = np.zeros((Lx, Ly))

    for x, y, v in zip(x_indx, y_indx, value):
        values[x, y] = v
    
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(values)

    ft = np.abs(np.fft.fft(values))
    ft = np.fft.fftshift(ft)
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1, 1, 1)
    ax2.imshow(ft)
    plt.show()

if __name__ == "__main__":
    for arg in sys.argv:
        if "--help" in arg:
            print(USAGE_MSG)
            exit(0)
    
    main(sys.argv[1])
