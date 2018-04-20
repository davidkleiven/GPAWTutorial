import json
import numpy as np
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
from scipy import signal

def plot_hist( data ):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for dset in data:
        x = dset["histogram"]["size"]
        hist = np.array( dset["histogram"]["occurence"] )
        print ("Number of entries in histogram: {}".format(np.sum(hist)))
        hist /= np.sum(hist)
        ax.plot( x, hist, ls="steps", label="{}K".format(dset["temperature"]) )
    ax.set_xlabel( "Cluster size" )
    ax.set_ylabel( "Relative occurence" )
    ax.set_yscale("log")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    yticks = ax.get_yticks()
    yticklabels = ["\$10^{%d}\$"%(int(np.log10(num))) for num in yticks]
    ax.set_yticklabels(yticklabels)

def main():
    fnames = ["data/cluster_statistics_50_200K.json","data/cluster_statistics_50_250K.json",\
    "data/cluster_statistics_50_300K.json","data/cluster_statistics_50_400K.json"]
    data = []
    for fname in fnames:
        try:
            with open(fname,'r') as infile:
                dset = json.load(infile)

            data.append(dset)
        except IOError as exc:
            print (str(exc))
            print ("Skipping this file")


    plot_hist(data)
    plt.show()

if __name__ == "__main__":
    main()
