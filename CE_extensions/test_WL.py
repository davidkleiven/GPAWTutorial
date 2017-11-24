from ase.calculators.emt import EMT
from ase.build import bulk
from wang_landau_scg import WangLandauSGC
from matplotlib import pyplot as plt
import pickle as pkl

def plot_probablility_distribution(wl):
    T = [100,200,300,400,500,800,1000,1200]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for temp in T:
        ax.plot( wl.E, wl.dos*wl._boltzmann_factor(temp), ls="steps", label="T={} K".format(temp) )
    ax.legend()
    ax.set_yscale("log")
    return fig

def main():
    new_run = False
    atoms = bulk("Ag")
    atoms = atoms*(4,4,4)
    calc = EMT()
    atoms.set_calculator(calc)
    chem_pot = {"Ag":0.0,"Au":0.0}
    atoms[0].symbol = "Au"
    site_types = [0 for _ in range(len(atoms))]
    site_elements = [["Ag","Au"]]
    if ( new_run ):
        wl = WangLandauSGC( atoms, calc, chemical_potentials=chem_pot, site_types=site_types, site_elements=site_elements, Nbins=50 )
    else:
        with open("wang_landau_ag_au.pkl", "rb") as infile:
            wl = pkl.load(infile)
    wl.run( maxsteps=1000 )
    #wl.set_number_of_bins(40)
    wl.save("wang_landau_ag_au.pkl")
    wl.plot_dos()
    wl.plot_histogram()
    plot_probablility_distribution(wl)
    plt.show()

if __name__ == "__main__":
    main()
