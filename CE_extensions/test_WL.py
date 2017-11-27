from ase.calculators.emt import EMT
from ase.build import bulk
from wang_landau_scg import WangLandauSGC
from matplotlib import pyplot as plt
import pickle as pkl
from sgc_to_cannonical import SGCToCanonicalConverter
import sys

def plot_probablility_distribution(wl):
    T = [100,200,300,400,500,800,1000,1200]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for temp in T:
        ax.plot( wl.E, wl.dos*wl._boltzmann_factor(temp), ls="steps", label="T={} K".format(temp) )
    ax.legend()
    ax.set_yscale("log")
    return fig

def analyze():
    wls = ["wang_landau_ag_au_converged_0.pkl", "wang_landau_ag_au_converged_8.pkl","wang_landau_ag_au_16.pkl",
    "wang_landau_ag_au_24.pkl", "wang_landau_ag_au_32.pkl", "wang_landau_ag_au_064.pkl"]
    wl_objs = []
    for fname in wls:
        with open(fname,'rb') as infile:
            wl_objs.append( pkl.load(infile) )
    analyzer = SGCToCanonicalConverter(wl_objs)
    T = [100,200,300,400,500,800,1000]
    chem_pot, comp, sgc_pots, chem_pot_raw, sgc_pots_raw = analyzer.get_compositions(T[2])
    print (chem_pot)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( chem_pot["Ag"], sgc_pots["Ag"], label="Free energy" )
    ax.plot( chem_pot_raw["Ag"], sgc_pots_raw["Ag"], 'o', mfc="none" )
    ax.set_xlabel( "Chemical potential" )
    ax.set_ylabel( "SGC potential" )
    ax2 = ax.twinx()
    ax2.plot( chem_pot["Ag"], comp["Ag"], color="#fdbf6f" )
    ax.plot( [],[], color="#fdbf6f", label="Conc")
    ax2.set_ylabel( "Concentration Ag" )
    ax.legend( loc="best", frameon=False )
    plt.show()

    print (chem_pot)

def main():
    new_run = False
    atoms = bulk("Ag")
    atoms = atoms*(4,4,4)
    calc = EMT()
    atoms.set_calculator(calc)
    chem_pot = {"Ag":0.064/64,"Au":0.0}
    atoms[0].symbol = "Au"
    site_types = [0 for _ in range(len(atoms))]
    site_elements = [["Ag","Au"]]
    if ( new_run ):
        wl = WangLandauSGC( atoms, calc, chemical_potentials=chem_pot, site_types=site_types, site_elements=site_elements, Nbins=50 )
    else:
        with open("wang_landau_ag_au_064.pkl", "rb") as infile:
            wl = pkl.load(infile)
    wl.run( maxsteps=10000 )
    #wl.set_number_of_bins(40)
    wl.save("wang_landau_ag_au_064.pkl")
    wl.plot_dos()
    wl.plot_histogram()
    plot_probablility_distribution(wl)
    plt.show()

if __name__ == "__main__":
    opt = sys.argv[1]
    if ( opt == "analyze" ):
        analyze()
    elif ( opt == "run" ):
        main()
