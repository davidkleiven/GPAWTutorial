from cemc.mfa.mean_field_approx import MeanFieldApprox
from ase.io import read
import dill as pck
from ase.calculators.cluster_expansion.cluster_expansion import ClusterExpansion
from ase.units import kB
import numpy as np
from matplotlib import pyplot as plt

def main():
    with open( "data/bc_10x10x10.pkl", 'rb' ) as infile:
        bc,cf,eci = pck.load(infile)
    calc = ClusterExpansion( bc, cluster_name_eci=eci, init_cf=cf, logfile=None )
    bc.atoms.set_calculator(calc)
    chem_pot= {"c1_1":-1.072}
    mf = MeanFieldApprox( bc )
    T0 = 20.0
    T1 = 500.0
    beta = np.linspace( 1.0/(kB*T0), 1.0/(kB*T1), 100 )

    G = mf.free_energy( beta, chem_pot=chem_pot )

    # Plot results
    fig = plt.figure()
    ax_eng = fig.add_subplot(1,1,1)
    ax_eng.plot( 1.0/(kB*beta), G )
    ax_eng.set_xlabel( "Temperature (K)")
    ax_eng.set_ylabel( "Free energy (kJ/mol)" )
    plt.show()
if __name__ == "__main__":
    main()
