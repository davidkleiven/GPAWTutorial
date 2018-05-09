import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
import numpy as np
from ase.db import connect
import json
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from cemc.tools import CanonicalFreeEnergy
from matplotlib import pyplot as plt
plt.switch_backend("TkAgg")

mc_db_name = "data/almgsi_fixed_composition800to100.db"
#fig = mlab.figure()

def get_free_energies():
    db = connect( mc_db_name )
    unique_formulas = []
    for row in db.select():
        if ( row.formula not in unique_formulas ):
            unique_formulas.append(row.formula)

    result = {}
    for formula in unique_formulas:
        internal_energy = []
        temperature = []
        conc = None
        for row in db.select( formula=formula ):
            if ( conc is None ):
                conc = {}
                for key,value in row.count_atoms().iteritems():
                    conc[key] = float(value)/row.natoms
            internal_energy.append( row.internal_energy/row.natoms )
            temperature.append( row.temperature )
        free_eng = CanonicalFreeEnergy(conc)
        temp,internal,F = free_eng.get( temperature, internal_energy )
        result[formula] = {}
        result[formula]["temperature"] = temp
        result[formula]["internal_energy"] = internal
        result[formula]["free_energy"] = F
        result[formula]["entropy"] = (internal-F)/temp
        result[formula]["TS"] = internal-F
        result[formula]["conc"] = conc
    return result

def excess():
    ref_energies = {
        "Mg":-1.599,
        "Si":-4.864,
        "Al":-3.737
    }
    res = get_free_energies()

    temp_indx = [0]
    surf = None
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for indx in temp_indx:
        mg_concs = []
        si_concs = []
        form_energy = []
        for formula,data in res.iteritems():
            if ( "Mg" in data["conc"].keys() ):
                mg_concs.append(data["conc"]["Mg"])
            else:
                mg_concs.append(0.0)
            if ( "Si" in data["conc"].keys() ):
                si_concs.append(data["conc"]["Si"])
            else:
                si_concs.append( 0.0 )
            form = data["internal_energy"][indx]
            for key,value in ref_energies.iteritems():
                if ( key in data["conc"].keys() ):
                    form -= data["conc"][key]*value
            form_energy.append( form )
        #mg_concs.append(0.0)
        #si_concs.append(0.0)
        #form_energy.append(0.0)
        mg_concs.append(1.0)
        si_concs.append(0.0)
        form_energy.append(0.0)
        mg_concs = np.array(mg_concs)
        si_concs = np.array(si_concs)
        x = mg_concs/(mg_concs+si_concs)
        srt_indx = np.argsort(x).astype(int)
        x = [x[indx] for indx in srt_indx]
        form_energy = [form_energy[indx] for indx in srt_indx]
        ax.plot(x,form_energy, "o", mfc="none")
        add_covnex_hull(x,form_energy,ax)
    plt.show()

def add_covnex_hull( x, y, ax, color="black" ):
    """
    Plots the convex hull
    """
    hull = ConvexHull( np.vstack((x,y)).T )
    for simplex in hull.simplices:
        print (simplex)
        if ( y[simplex[0]] <= 0.0 and y[simplex[1]] <= 0.0 ):
            ax.plot( [ x[simplex[0]],x[simplex[1]] ], [ y[simplex[0]],y[simplex[1]] ], color=color )

if __name__ == "__main__":
    excess()
    #fig.scene.render()
    #mlab.show()
