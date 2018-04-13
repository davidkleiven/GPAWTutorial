import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
import json

def phase_diag_extremal():
    fname = "data/extremal_points.json"
    with open(fname,'r') as infile:
        data = json.load(infile)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    al_line = []
    mg_line = []
    al3mg = []
    almg = []
    almg3 = []
    temps = []
    for entry in data:
        if ( len(entry["minima"]) == 0 ):
            continue
        temps.append(entry["temperature"])
        al_line.append( entry["maxima"][0] )
        mg_line.append( entry["maxima"][-1] )
        al3mg.append( entry["minima"][0] )
        almg.append( entry["minima"][1] )
        almg3.append( entry["minima"][2] )

    ax.plot( al_line, temps, marker="x", color="black" )
    ax.plot( mg_line, temps, marker="x", color="black" )
    ax.plot( al3mg, temps, marker="x", color="black" )
    ax.plot( almg, temps, marker="x", color="black" )
    ax.plot( almg3, temps, marker="x", color="black" )
    ax.set_xlabel( "Mg concentration" )
    ax.set_ylabel( "Temperature (K)" )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig

def main():
    phase_diag_extremal()
    plt.show()

if __name__ == "__main__":
    main()
