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
from ase.visualize import view


db_name = "bulk_modulus_fcc.db"
bc_fname = "bc_almg_fcc.pkl"
def eval_eci():
    with open(bc_fname,'rb') as infile:
        bc = pck.load(infile)

    print (bc.atoms.get_cell())
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
    formula = "Al"
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
    bm.set_average_mass( count )
    bm.fit()
    fig = bm.plot()
    V_fit = np.linspace(0.95*np.min(V), 1.05*np.max(V),100)
    B = bm.bulk_modulus( V )
    #E0,V0 = bm.minimum_energy()
    ax = fig.gca().twinx()

    ax.plot( V, B/GPa, color="#fc8d62", marker="x" )
    ax.set_xlabel( "Volume (\$\SI{}{\\angstromg^3}\$)" )
    ax.set_ylabel( "Bulk modulus (GPa)" )

    # Compute the temperature volume curve
    T = np.array( [200,293,300,400,500,600,700,800] )
    vols = bm.volume_temperature( T, natoms )
    print ("Debye temperature {}: {}K".format(formula,bm.debye_temperature(vols[1])))
    print ("Bulk modulus {}: {}GPa".format(formula,bm.bulk_modulus(vols[1])/GPa))
    print ("Speed of sound {}: {}m/s".format(formula,bm.speed_of_sound(vols[1]) ))
    print ("Density {}: {}g/cm^3".format(formula,bm.density_g_per_cm3(vols[1])))
    alpha_L = bm.linear_thermal_expansion_coefficient(T,vol_curve=vols)
    print ("Thermal expansion coefficient {}: {} K^-1".format( formula, alpha_L[1]) )
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
