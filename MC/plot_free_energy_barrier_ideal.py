import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import numpy as np
from ase.units import kB, kJ, mol
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress

n_atoms = 15**3
energy_file = "data/cluster_struct_new_trialmove/energies_run2.txt"

ref_en_al = -239.147/64.0 # DFT value
ref_en_mg = -101.818/64.0 # DFT value

ref_en_al = -3.73659854515 # CE value
ref_en_mg = -1.59155450937 # CE value

en_al63mg = -236.921
en_al63mg_form = en_al63mg - (63*ref_en_al+ref_en_mg)

en_solute = en_al63mg/64.0
en_solute_form= en_al63mg_form/64.0
x = 0.1

def extract_local_maxima(array,show=False):
    loc_max = []
    x = range(0,len(array))
    plt.plot(x,array,'o')
    spl = UnivariateSpline( x, array, k=2, s=0 )
    x = np.linspace(0.0,np.max(x),1000)
    y = spl(x)
    indices = []
    if ( show ):
        plt.plot(x,y)
        plt.show()
    for i in range(1,len(x)-1):
        if ( spl(x[i]) > spl(x[i-1]) and spl(x[i]) > spl(x[i+1]) ):
            loc_max.append(x[i])
            indices.append(i)
    return loc_max,spl(loc_max)

def critical_size( dH, size ):
    T = np.linspace(273,373,10)
    kbtx = kB*np.log(0.1)*T
    nmax = []
    barrier = []
    for i in range(len(T)):
        F = dH - kbtx[i]*(size-1)
        loc_max,fvals = extract_local_maxima(F,show=True)
        print (loc_max)
        if ( len(loc_max) > 0 ):
            indx = np.argmax(fvals)
            nmax.append(loc_max[indx])
            barrier.append(fvals[indx])
        else:
            nmax.append(-1)
            barrier.append(0)
    return T,nmax,barrier



def main():
    size,energy = np.loadtxt( energy_file, delimiter=",", unpack=True )
    energy -= (size*ref_en_mg + (n_atoms-size)*ref_en_al)

    #energy -= en_al63mg_form*size
    size_temp = np.linspace(0,np.max(size),len(size))
    dilution_limit = (energy[1]-energy[0])*size_temp

    dH = energy - size*(energy[1]-energy[0])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( size, energy, marker="o", mfc="none" )
    ax.plot( size_temp, dilution_limit )
    ax.set_xlabel( "Cluster size" )
    ax.set_ylabel( "Cluster energy (eV)" )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    T = [353,293]
    concs = [0.05,0.1,0.15]
    gw_kw = {"hspace":0.1}
    fig_F,axF = plt.subplots(nrows=2,sharex=True,gridspec_kw=gw_kw)
    axF[1].axhline( 0, ls="--", color="grey")
    axF[0].axhline( 0, ls="--", color="grey")
    #axF = fig_F.add_subplot(1,1,1)
    for i in range(2):
        for conc in concs:
            dS = kB*(size-1)*np.log(conc)
            dS[0] = 0.0
            F = dH - T[i]*dS
            axF[i].plot( size, F, marker="o", mfc="none",label="\${}\%\$".format(int(conc*100)) )
        axF[i].spines["right"].set_visible(False)
    axF[0].spines["top"].set_visible(False)
    axF[1].spines["top"].set_visible(False)
    axF[0].legend( loc="best", frameon=False, ncol=3, columnspacing=0.05 )
    #axF[0].spines["bottom"].set_position("zero")
    #axF[1].spines["bottom"].set_position("zero")
    axF[0].text( 5,100, "T=353K")
    axF[1].text( 5,50, "T=293K")
    axF[1].set_xlabel( "Cluster size" )
    axF[0].set_ylabel( "Free energy (eV)")
    #axF[0].set_xticklabels([])

    dH_start_one = dH[1:]*1000.0
    size_start_one = size[1:]
    dH_per_atom = dH_start_one/size_start_one
    slope, interscept, rvalue, pvalue, stderr = linregress( size_start_one[1:]**(-1.0/3.0), dH_per_atom[1:] )
    n_fit = np.linspace( 1.0, 2.0*np.max(size), 10 )
    dH_fit = interscept + slope*n_fit**(-1.0/3.0)
    fig_shape = plt.figure()
    ax_shape = fig_shape.add_subplot(1,1,1)
    ax_shape.plot( size_start_one**(-1.0/3.0), dH_per_atom, "o", mfc="none")
    ax_shape.plot( n_fit**(-1.0/3.0), dH_fit )
    ax_shape.set_ylabel( "Normalized Cluster Energy (meV/atom)")
    ax_shape.set_xlabel( "\$1/\sqrt[3]{N}\$" )
    ax_shape.spines["right"].set_visible(False)
    ax_shape.spines["top"].set_visible(False)
    print ("Slope: {}. interscept: {}".format(slope,interscept))
    plt.show()

if __name__ == "__main__":
    main()
