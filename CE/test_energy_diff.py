import gpaw as gp
from ase.build import bulk
from ase.optimize import BFGS
from ase.constraints import UnitCellFilter
from matplotlib import pyplot as plt
import numpy as np

def run_dft():
    kpt = 16
    atoms = bulk("Al",crystalstructure="fcc",a=4.05)
    calc = gp.GPAW( mode=gp.PW(600), xc="PBE", kpts=(kpt,kpt,kpt), nbands=-50 )
    atoms.set_calculator( calc )
    relaxer = BFGS( UnitCellFilter(atoms) )
    relaxer.run( fmax=0.025 )
    energy = atoms.get_potential_energy()
    print (energy)

def plot_results():
    fname = "data/almg_energy_diff_test.csv"
    kpt, al, mg = np.loadtxt(fname, delimiter=",", unpack=True )

    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    ax.plot( kpt, al, label="Al" )
    ax.plot( kpt, mg, label="Mg" )
    avg = 0.5*(al+mg)
    ax2 = fig.add_subplot(2,1,2)
    ax2.plot(kpt,al-mg)
    ax2.set_xlabel( "Number of k-points" )
    ax2.set_ylabel( "$E_{al} - E_{mg}$ (eV/atom)" )
    ax.set_ylabel( "Energy (eV)")
    ax.legend()
    plt.show()

if __name__ == "__main__":
    #plot_results()
    run_dft()
