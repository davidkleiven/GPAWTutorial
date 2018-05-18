import json
import numpy as np
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
mpl.rcParams["svg.fonttype"] = "none"
from matplotlib import pyplot as plt
from ase.units import kJ, mol

temps = [200,300,400]
colors = ["#8c2d04","#ec7014","#fec44f"]
colors = ['#a6cee3','#1f78b4','#b2df8a']
colors = ['#7fc97f','#beaed4','#fdc086']
colors = ['#e41a1c','#377eb8','#4daf4a']
colors = ['#1b9e77','#d95f02','#7570b3']

ref_al = -3.73667187
ref_mg = -1.59090625

def plot_fcc( ax ):
    fname = "data/free_energies_fcc.json"
    with open(fname,'r') as infile:
        data = json.load(infile)
    has_legend = False
    for i,T in enumerate(temps):
        concs = []
        F = []
        for formula,entries in data.iteritems():
            data_T = np.array( entries["temperature"] )
            indx = np.argmin( np.abs(data_T - T) )

            mg_conc = entries["conc"]["Mg"]
            energy = entries["free_energy"][indx] - mg_conc*ref_mg - (1.0-mg_conc)*ref_al
            F.append( energy )
            concs.append(mg_conc)

        srt_indx = np.argsort(concs)
        concs = [concs[indx] for indx in srt_indx]
        F = np.array( [F[indx] for indx in srt_indx] )

        ax.plot( concs, F*mol/kJ, marker="o", mfc="none", color=colors[i], label="{}K".format(T) )

def plot_217( ax ):
    fname = "data/free_energies_217.json"
    fname = "data/almg_217mfs_free_eng.json"
    with open(fname,'r') as infile:
        data = json.load(infile)
    has_legend = False
    for i,T in enumerate(temps):
        concs = []
        F = []
        for formula,entries in data.iteritems():
            data_T = np.array( entries["temperature"] )
            indx = np.argmin( np.abs(data_T - T) )
            mg_conc = entries["conc"]["Mg"]
            #print (len(entries["free_energy"]))
            try:
                energy = entries["free_energy"][indx]# - mg_conc*ref_mg - (1.0-mg_conc)*ref_al
                F.append( energy )
                concs.append(mg_conc)
            except Exception as exc:
                print (str(exc))

        srt_indx = np.argsort(concs)
        concs = [concs[indx] for indx in srt_indx]
        F = np.array( [F[indx] for indx in srt_indx] )

        ax.plot( concs, F*mol/kJ, marker="D", mfc="none", color=colors[i] )

def plot_hcp( ax ):
    concs = [0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95]

    for i,T in enumerate(temps):
        F = []
        for conc in concs:
            fname = "data/mc_hcp/free_eng/freeenergy_{}.csv".format(conc)
            temperature, free_energy, mg_conc = np.loadtxt( fname, delimiter=",", unpack=True, usecols=(0,1,3) )
            indx = np.argmin( np.abs(temperature-T) )
            energy = free_energy[indx] - mg_conc[0]*ref_mg - (1.0-mg_conc[0])*ref_al
            F.append( energy )
        F = np.array(F)
        ax.plot( concs, F*mol/kJ, marker="^", mfc="none", ls="--", color=colors[i] )

def main():
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    plot_fcc(ax)
    plot_hcp(ax)
    plot_217(ax)
    ax.legend(frameon=False)
    ax.set_xlabel( "Mg concentration" )
    ax.set_ylabel( "Free energy of formation (kJ/mol)" )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
