import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
import numpy as np
from ase.db import connect
import json
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from cemc.tools import CanonicalFreeEnergy
from matplotlib import pyplot as plt
from ase.units import kJ, mol
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d, Axes3D
plt.switch_backend("TkAgg")

mc_db_name = "data/almgsi_fixed_composition.db"
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

    temp_indx = [-1]
    surf = None
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    figsurf = plt.figure()
    axsurf = Axes3D(figsurf)
    figcont = plt.figure()
    axcont = figcont.add_subplot(1,1,1)
    figscat = plt.figure()
    axscat = figscat.add_subplot(1,1,1)
    for indx in temp_indx:
        mg_concs = []
        si_concs = []
        form_energy = []
        formulas = []
        for formula,data in res.iteritems():
            formulas.append(formula)
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
        formulas.append("Mg1000")
        mg_concs.append(0.0)
        si_concs.append(0.0)
        form_energy.append(0.0)
        formulas.append("Al1000")
        mg_concs = np.array(mg_concs)
        si_concs = np.array(si_concs)
        x = mg_concs/(mg_concs+si_concs)
        x = mg_concs
        srt_indx = np.argsort(x).astype(int)
        x = [x[indx] for indx in srt_indx]
        mg_concs = np.array(x)
        si_concs = [si_concs[indx] for indx in srt_indx]
        formulas = [formulas[indx] for indx in srt_indx]
        form_energy = [form_energy[indx] for indx in srt_indx]
        form_energy = np.array(form_energy)*mol/kJ
        unique_si_concs = np.unique( si_concs )
        cm = plt.cm.get_cmap('viridis')
        convex_hull3D( mg_concs, si_concs, form_energy, formulas )
        for c_si in unique_si_concs:
            c_mg = mg_concs[si_concs==c_si]
            e_form = form_energy[si_concs==c_si]
            forms = np.array(formulas)[si_concs==c_si]
            srt = np.argsort(c_mg)
            c_mg = [c_mg[indx] for indx in srt]
            e_form = [e_form[indx] for indx in srt]
            print (c_mg)

            scaled_si_conc = (c_si-np.min(si_concs))/(np.max(si_concs)-np.min(si_concs))
            color=cm(scaled_si_conc)
            print (scaled_si_conc)
            ax.plot(c_mg, e_form, marker="o", mfc="none", color=color )
            #add_covnex_hull( c_mg, e_form, ax, color=color, formulas=forms )

        #cbar = fig.colorbar(sc)
        #cbar.set_label("Si concentration")
        #add_covnex_hull(x,form_energy,ax,formulas=formulas)
        ax.set_xlabel( "Mg concentration")
        ax.set_ylabel( "Enthalpy of formation (kJ/mol)" )

        pts = np.vstack( (mg_concs,si_concs) ).T
        xmg = np.linspace( 0.0, 1.0, 200 )
        xsi = np.linspace(0.0,0.3, 200 )
        xmg,xsi = np.meshgrid( xmg, xsi )
        E = griddata( pts, form_energy, (xmg,xsi), fill_value=0.0, method="linear" )
        axsurf.plot_surface( xmg, xsi, E, cmap="inferno" )
        im = axcont.contourf( xmg, xsi, E, 256, cmap="RdYlBu_r" )
        cbar = figcont.colorbar(im)
        cbar.set_label( "Enthalpy of Formation (kJ/mol)" )
        axcont.set_xlabel( "Mg concentration" )
        axcont.set_ylabel( "Si concentration" )
        axscat.scatter( mg_concs, si_concs, c=form_energy, s=50 )
    plt.show()

def convex_hull3D( mg_conc, si_conc, E, formulas ):
    hull = ConvexHull( np.vstack((mg_conc,si_conc,E)).T )
    printed = []
    for simplex in hull.simplices:
        for s in simplex:
            if ( formulas[s] not in printed ):
                print ( "{}: {} kJ/mol".format(formulas[s],E[s]) )
                printed.append(formulas[s])
        #print ( "{}: {} kJ/mol".format(formulas[simplex[1]],E[simplex[1]]) )

def add_covnex_hull( x, y, ax, color="black", formulas=None ):
    """
    Plots the convex hull
    """
    hull = ConvexHull( np.vstack((x,y)).T )
    slope = (y[-1]-y[0])/(x[-1]-x[0])
    for i,simplex in enumerate(hull.simplices):
        diff = np.abs(x[simplex[0]] - x[simplex[1]])
        if ( diff > 0.6 ):
            continue
        if ( formulas is not None ):
            ax.text( x[simplex[0]], y[simplex[0]], formulas[simplex[0]])
            ax.text( x[simplex[1]], y[simplex[1]], formulas[simplex[1]])
        if ( y[simplex[0]] <= 0.0 and y[simplex[1]] <= 0.0 ):
            ax.plot( [ x[simplex[0]],x[simplex[1]] ], [ y[simplex[0]],y[simplex[1]] ], color=color )


if __name__ == "__main__":
    excess()
    #fig.scene.render()
    #mlab.show()
