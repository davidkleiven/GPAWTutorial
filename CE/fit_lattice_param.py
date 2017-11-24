import numpy as np
from scipy.stats import linregress
from matplotlib import pyplot as plt

def main():
    fname = "almg_lattice_parameter.csv" # From J. L. Murray, The Al-Mg system, 1982
    data = np.loadtxt( fname, delimiter=",")

    mg_conc = data[:,0]
    lattice_param = data[:,1]*10

    slope, interscept, r_value, p_value, stderr = linregress( mg_conc, lattice_param )

    mg_conc_fit = np.linspace( 0.0, 0.5 )
    fitted_lat_par = interscept+slope*mg_conc_fit
    print ("Interscept: %.4E"%(interscept))
    print ("Slope: %.4E"%(slope))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( mg_conc, lattice_param, 'o', mfc="none")
    ax.plot( mg_conc_fit, fitted_lat_par )
    plt.show()

if __name__ == "__main__":
    main()
