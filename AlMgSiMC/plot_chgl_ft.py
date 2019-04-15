from matplotlib import pyplot as plt
import numpy as np

def plot_ft(fname):
    data = []

    # with open(fname, 'r') as infile:
    #     for line in infile:
    #         splitted = line.split(",")[:-1]
    #         data = [float(v) for v in splitted]

    data = np.loadtxt(fname, delimiter=",")

    L = int(np.sqrt(len(data)))
    data = np.reshape(data, (L, -1))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(np.log(data))
    plt.show()

plot_ft("data/conc_ft.csv")