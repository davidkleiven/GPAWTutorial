import numpy as np
from matplotlib import pyplot as plt
import sys


def show(fname):
    with open(fname, 'r') as infile:
        line = infile.readlines()[0]
        data = line.split(",")[:-1]
    data = list(map(lambda x: float(x), data))
    data = np.array(data)
    N = len(data)
    Nx = int(np.sqrt(N))
    print(Nx)
    data = data.reshape((Nx, -1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(data, origin="lower")
    plt.show()

if __name__ == "__main__":
    show(sys.argv[1])