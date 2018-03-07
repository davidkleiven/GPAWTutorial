from atomtools.eos.birch_murnagan import BirschMurnagan
from ase.db import connect
import pickle as pck
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
import numpy as np
from ase.units import GPa
from scipy.stats import linregress
from atomtools.ce.phonon_ce_eval import PhononEvalEOS


db_name = "bulk_modulus_fcc.db"
bc_fname = "bc_almg_fcc.pkl"
def eval_eci():
    with open(bc_fname,'rb') as infile:
        bc = pck.load(infile)

    temperatures = [100,200,300,400,500,600,700,800]
    evaluator = PhononEvalEOS( bc, db_name, penalty=None, cluster_names=["c0","c1_1","c2_707_1_1"] )
    ecis = {"c1_1":[],"c2_707_1_1":[]}
    for T in temperatures:
        evaluator.temperature = T
        eci_name = evaluator.get_cluster_name_eci_dict
        ecis["c1_1"].append( eci_name["c1_1"])
        ecis["c2_707_1_1"].append( eci_name["c2_707_1_1"] )
    #evaluator.plot_energy()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( temperatures, ecis["c2_707_1_1"], label="c2", marker="x" )
    ax.plot( temperatures, ecis["c1_1"], label="singl", marker="o" )
    ax.set_xlabel( "Temperature (K)" )
    ax.set_ylabel( "ECI/\$k_BT\$" )
    ax.legend( loc="best", frameon=False )
    plt.show()

def main():
    db = connect( db_name )
    formula = "Al2Mg6"
    V = []
    E = []
    natoms = None
    count = {}
    for row in db.select( formula=formula ):
        V.append( row.volume )
        E.append( row.energy )
        natoms = row.natoms
        count = row.count_atoms()

    V = np.array(V)
    E = np.array(E)

    bm = BirschMurnagan(V,E)
    bm.fit()
    fig = bm.plot()
    V_fit = np.linspace(0.95*np.min(V), 1.05*np.max(V),100)
    B = bm.bulk_modulus( V )
    ax = fig.gca().twinx()

    ax.plot( V, B/GPa, color="#fc8d62", marker="x" )
    ax.set_xlabel( "Volume (\$\SI{}{\\angstromg^3}\$)" )
    ax.set_ylabel( "Bulk modulus (GPa)" )

    # Compute the temperature volume curve
    T = np.array( [200,300,400,500,600,700,800] )
    bm.set_average_mass( count )
    vols = bm.volume_temperature( T, natoms )
    alpha_L = bm.linear_thermal_expansion_coefficient(T,vol_curve=vols)
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot( T, vols, marker="x", color="#80b1d3" )
    ax3 = ax2.twinx()
    ax3.plot( T, alpha_L*1E6, marker="o", color="#fb8072" )
    ax3.set_ylabel( "Linear thermal expansion (\$10^{-6}\$ K\$^{-1}\$)")
    ax2.set_ylabel( "Volume (\$\SI{\\angstrom^3}\$)")
    ax2.set_xlabel( "Temperature (K)" )
    plt.show()

if __name__ == "__main__":
    eval_eci()
    #main()
