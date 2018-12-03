import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import numpy as np
from ase.units import kB, kJ, mol
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress
import json

n_atoms = 15**3
energy_file = "data/cluster_struct_new_trialmove/energies_run2.txt"
FREE_ENG_FILE = "data/almg_review/free_energy_size.json"
#energy_file = "data/cluster_struct_new_trialmove/energies_backup.txt"

ref_en_al = -239.147/64.0 # DFT value
ref_en_mg = -101.818/64.0 # DFT value

ref_en_al = -3.73659854515 # CE value
ref_en_mg = -1.59155450937 # CE value

ref_en_al = -3.7365039196426015 # CE value
ref_en_mg = -1.5911398128049632 # CE value

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
        if ( len(loc_max) > 0 ):
            indx = np.argmax(fvals)
            nmax.append(loc_max[indx])
            barrier.append(fvals[indx])
        else:
            nmax.append(-1)
            barrier.append(0)
    return T,nmax,barrier

def barrier_vs_temp(dH=None, dF=None):
    concs = [0.01,0.05,0.1,0.15]
    T = np.linspace(100,400,40)
    size = np.array( range(len(dH)) )
    if dF is not None:
        T = [int(k) for k in dF.keys()]
        T = np.array(T)
        size = np.array([k for k in dF[T[0]].keys()])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    #ax_cs = ax.twinx()
    figbeta = plt.figure()
    axbeta = figbeta.add_subplot(1,1,1)
    
    colors = ["#5D5C61", "#878796", "#7395AE", "#557A95", "#B1A296"]
    for num, conc in enumerate(concs):
        dS = kB*(size-1)*np.log(conc)
        print(dS)
        dS[0] = 0.0
        barrier = []
        crit_size = []
        for i in range(len(T)):
            if dF is not None:
                F = np.array([v for k, v in dF[T[i]].items()])
                F -= T[i]*dS
            else:
                F = dH - T[i]*dS
            #F = dH[2:] - T[i]*dS
            barrier.append( np.max(F) )
            crit_size.append(np.argmax(F))
        #plt.plot(size, F)
        #plt.show()
        crit_size = np.array(crit_size)
        #print(crit_size)
        #print(conc, crit_size, T)
        barrier = np.array(barrier)
        ax.plot(T[crit_size<45], barrier[crit_size<45], label="{}%".format(int(100*conc)), marker="o", mfc="none", color=colors[num])
        #ax_cs.plot(T[crit_size<49], crit_size[crit_size<49], marker="v", mfc="none")
        beta = 1.0/(kB*T[crit_size<45])
        axbeta.plot( T[crit_size<45], beta*barrier[crit_size<45], label="\${}\%$".format(int(100*conc)), marker="o", mfc="none", color=colors[num])

    ax.legend(frameon=False)
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel( "Temperature (K)" )
    ax.set_ylabel( "Free energy barrier (eV)"  )
    axbeta.legend(frameon=False)
    axbeta.spines["right"].set_visible(False)
    axbeta.spines["top"].set_visible(False)
    axbeta.set_xlabel( "Temperature (K)" )
    axbeta.set_ylabel( "\$\\beta \Delta F\$" )


def main():
    size,energy = np.loadtxt( energy_file, delimiter=",", unpack=True )
    energy -= (size*ref_en_mg + (n_atoms-size)*ref_en_al)

    #energy -= en_al63mg_form*size
    size_temp = np.linspace(0,np.max(size),len(size))
    dilution_limit = (energy[1]-energy[0])*size_temp
    strain_per_atom = 5.417643731247179E-3
    #eshelby_strian = 4*strain_per_atom*size_temp

    dH = energy - size*(energy[1]-energy[0])

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.fill_between(size, 0, energy, color="#5D5C61", alpha=0.4)
    ax.plot(size, energy,color="#5D5C61")
    ax.plot(size_temp, dilution_limit, color="#7395AE")
    #ax.plot(size_temp, eshelby_strian, color="#557A95")
    ax.fill_between(size, energy, energy + 4*strain_per_atom*size, color="#557A95", alpha=0.4)
    ax.plot(size, energy + 4*strain_per_atom*size, color="#557A95")
    ax.set_xlabel("Num. formula units")
    ax.set_ylabel("Energy (eV)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)

    dH += 4*strain_per_atom*size

    T = [353,293]
    T = [300, 100]
    concs = [0.01, 0.05,0.1]
    gw_kw = {"hspace":0.1}
    fig_F,axF = plt.subplots(nrows=2,sharex=True,gridspec_kw=gw_kw)
    axF[1].axhline( 0, ls="--", color="grey")
    axF[0].axhline( 0, ls="--", color="grey")
    #axF = fig_F.add_subplot(1,1,1)
    colors = ["#5D5C61", "#878796", "#7395AE", "#557A95", "#B1A296"]
    dF_size_temp = {}
    for i in range(2):
        for num, conc in enumerate(concs):
            dS = kB*(size-1)*np.log(conc)
            dS[0] = 0.0
            F = dH - T[i]*dS
            print("Critical size {}".format(size[np.argmax(F)]))
            axF[i].plot( size, F, marker="o", mfc="none",label="\${}\%\$".format(int(conc*100)), color=colors[num] )
        axF[i].spines["right"].set_visible(False)
        axF[i].spines["right"].set_visible(False)

    data = None
    try:
        with open(FREE_ENG_FILE, 'r') as infile:
            data = json.load(infile)
        sizes_free_eng = np.array(data["sizes"])

        # Construct datastructure for free energy as a function of temperature and size
        for temperature, val in data["free_energy"].items():
            dF_size_temp[int(temperature)] = {size: val[i] - (size*ref_en_mg + (n_atoms-size)*ref_en_al) 
                                    + 4*strain_per_atom*size 
                                    - size*(energy[1]-energy[0]) 
                                    for  i, size in enumerate(data["sizes"])}
        for i, temp in enumerate(T):
            print(temp)
            F = np.array(data["free_energy"][str(temp)])
            dF = F - (sizes_free_eng*ref_en_mg + (n_atoms-sizes_free_eng)*ref_en_al)
            #dF = energy[-len(F):]
            dF -= sizes_free_eng*(energy[1]-energy[0])
            dF += 4*strain_per_atom*sizes_free_eng
            for num, conc in enumerate(concs):
                dS = kB*(sizes_free_eng-1)*np.log(conc)
                # dS[0] = 0.0
                #F = dH - T[i]*dS
                #dF -= T[i]*dS
                axF[i].plot(sizes_free_eng, dF-T[i]*dS, color=colors[num], ls="--" )
    except Exception as exc:
        print(str(exc)+type(exc).__name__)
    axF[0].spines["top"].set_visible(False)
    axF[1].spines["top"].set_visible(False)
    axF[0].legend( loc="best", frameon=False, ncol=3, columnspacing=0.05 )
    #axF[0].spines["bottom"].set_position("zero")
    #axF[1].spines["bottom"].set_position("zero")
    axF[0].text( 5,100, "T=293K")
    axF[1].text( 5,50, "T=120K")
    axF[1].set_xlabel( "Num. formula units" )
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
    if dF_size_temp != {}:
        barrier_vs_temp(dH=dH, dF=dF_size_temp)
    else:
        barrier_vs_temp(dH=dH)
    plt.show()

if __name__ == "__main__":
    main()
