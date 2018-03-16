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
import json


db_name = "bulk_modulus_fcc.db"
bc_fname = "data/BC_fcc.pkl"
def eval_eci():
    with open(bc_fname,'rb') as infile:
        bc = pck.load(infile)

    print (bc.cluster_names)
    print (bc.atoms.get_cell())
    print (bc.basis_functions)
    temperatures = [100,200,300,400,500,600,700,800]
    #cluster_names = ["c0","c1_0","c2_1000_1_00","c2_1155_1_00"]
    cluster_names = ["c0","c1_0","c2_1000_1_00"]
    evaluator = PhononEvalEOS( bc, db_name, penalty=None, cluster_names=cluster_names )
    ecis = {name:[] for name in cluster_names}
    for T in temperatures:
        evaluator.temperature = T
        eci_name = evaluator.get_cluster_name_eci_dict
        for key,value in eci_name.iteritems():
            ecis[key].append( value )
    print (ecis)
    evaluator.plot_energy()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for key,value in ecis.iteritems():
        ax.plot( temperatures, value, label=key, marker="x" )
    ax.set_xlabel( "Temperature (K)" )
    ax.set_ylabel( "ECI/\$k_BT\$" )
    ax.legend( loc="best", frameon=False )
    plt.show()

def compute_thermal_prop( gid ):
    db = connect( db_name )
    formula = "Mg"
    V = []
    E = []
    natoms = None
    count = {}
    mg_conc = 0.0
    for row in db.select( groupID=gid ):
        if ( row.get("energy") is None ):
            continue
        V.append( row.volume )
        E.append( row.energy )
        natoms = row.natoms
        count = row.count_atoms()
        formula = row.formula
        if ( "Mg" in count.keys() ):
            mg_conc = float(count["Mg"])/row.natoms

    V = np.array(V)
    E = np.array(E)

    bm = BirschMurnagan(V,E)
    bm.set_average_mass( count )
    V_fit = np.linspace(0.95*np.min(V), 1.05*np.max(V),100)
    B = bm.bulk_modulus( V )
    #E0,V0 = bm.minimum_energy()

    # Compute the temperature volume curve
    T = np.array( [200,293,300,400,500,600,700,800] )
    vols = bm.volume_temperature( T, natoms )
    T_D = bm.debye_temperature(vols[1])
    B_RT = bm.bulk_modulus(vols[1])/GPa
    c_sound = bm.speed_of_sound(vols[1])
    print ("Debye temperature {}: {}K".format(formula,T_D))
    print ("Bulk modulus {}: {}GPa".format(formula,B_RT))
    print ("Speed of sound {}: {}m/s".format(formula, c_sound))
    print ("Density {}: {}g/cm^3".format(formula,bm.density_g_per_cm3(vols[1])))
    alpha_L = bm.linear_thermal_expansion_coefficient(T,vol_curve=vols)
    print ("Thermal expansion coefficient {}: {} K^-1".format( formula, alpha_L[1]) )
    return mg_conc, B_RT, alpha_L[1], T_D, c_sound

def main():

    gid = [1,2,3,4,5,6]
    res = {"mg_conc":[],"debye_temp":[],"bulk_mod":[],"lin_thermal_exp":[]}
    for uid in gid:
        mg_conc,B,alpha_l,T_D, c_sound = compute_thermal_prop(uid)
        res["mg_conc"].append(mg_conc)
        res["bulk_mod"].append( B )
        res["lin_thermal_exp"].append( alpha_l )
        res["debye_temp"].append( T_D )

    with open( "data/thermal_prop_almg_fcc.json", 'w' ) as outfile:
        json.dump( res, outfile, sort_keys=True, indent=2, separators=(",",":") )

    """
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot( T, vols, marker="x", color="#80b1d3" )
    ax3 = ax2.twinx()
    ax3.plot( T, alpha_L*1E6, marker="o", color="#fb8072" )
    ax3.set_ylabel( "Linear thermal expansion (\$10^{-6}\$ K\$^{-1}\$)")
    ax2.set_ylabel( "Volume (\$\SI{\\angstrom^3}\$)")
    ax2.set_xlabel( "Temperature (K)" )
    plt.show()
    """

if __name__ == "__main__":
    eval_eci()
    #main()
