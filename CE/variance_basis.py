from atomtools.ce import PopulationVariance
import pickle as pck
from matplotlib import pyplot as plt

def main():
    fname = "data/BC_fcc.pkl" # File created by CEtesting.py
    with open( fname, 'rb' ) as infile:
        bc = pck.load( infile )

    popvar = PopulationVariance( bc )
    cov, mu = popvar.estimate( n_probe_structures=200 )
    eigval, eigvec = popvar.diagonalize( cov, plot=True )
    popvar.plot_eigen( eigval, eigvec )
    plt.show()

if __name__ == "__main__":
    main()
