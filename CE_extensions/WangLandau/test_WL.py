import sys
from ase.calculators.emt import EMT
from ase.build import bulk
from wang_landau_scg import WangLandauSGC
from matplotlib import pyplot as plt
import pickle as pkl
from sgc_to_cannonical import SGCToCanonicalConverter
from wang_landau_db_manger import WangLandauDBManger
import numpy as np

db_name = "/home/davidkl/Documents/GPAWTutorial/CE_extensions/WangLandau/wang_landau_ag_au_fixed_f.db"
def plot_probablility_distribution(wl):
    T = [100,200,300,400,500,800,1000,1200]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for temp in T:
        ax.plot( wl.E, wl.dos*wl._boltzmann_factor(temp), ls="steps", label="T={} K".format(temp) )
    ax.legend()
    ax.set_yscale("log")
    return fig

def initialize_db():
    manager = WangLandauDBManger( db_name )
    ag_chem_pot = np.linspace( 0.0,0.32/64, 20 )
    for i in range(len(ag_chem_pot) ):
        chem_pot = {
        "Ag":ag_chem_pot[i],
        "Au":0.0
        }
        manager.insert( chem_pot, initial_f=1.6, Nbins=100 )

def update_groups():
    manager = WangLandauDBManger( db_name )
    for group in range(manager.get_new_group()):
        for _ in range(20):
            manager.add_run_to_group(group)


def analyze():
    db_manager = WangLandauDBManger( db_name )
    analyzers = db_manager.get_analyzer_all_groups()
    analyzers[-1].plot_dos()
    analyzer = SGCToCanonicalConverter(analyzers,64)
    T = [100,200,300,400,500,800,1000]
    chem_pot, comp, sgc_pots, chem_pot_raw, sgc_pots_raw = analyzer.get_compositions(T[0])
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

def main( runID ):
    atoms = bulk("Ag")
    atoms = atoms*(4,4,4)
    calc = EMT()
    atoms.set_calculator(calc)
    chem_pot = {"Ag":0.064/64,"Au":0.0}
    atoms[0].symbol = "Au"
    site_types = [0 for _ in range(len(atoms))]
    site_elements = [["Ag","Au"]]
    wl = WangLandauSGC( atoms, calc, db_name, runID, site_types=site_types, site_elements=site_elements, Nbins=100, scheme="fixed_f", conv_check="histstd" )
    wl.run( maxsteps=100000 )
    wl.save_db()
    #wl.plot_dos()
    #wl.plot_histogram()
    #wl.plot_growth_fluctuation()
    #plot_probablility_distribution(wl)
    #plt.show()

if __name__ == "__main__":
    opt = sys.argv[1]
    if ( opt == "analyze" ):
        analyze()
    elif ( opt == "run" ):
        runID = int(sys.argv[2] )
        main(runID)
    elif ( opt == "init" ):
        initialize_db()
    elif ( opt == "add" ):
        update_groups()
