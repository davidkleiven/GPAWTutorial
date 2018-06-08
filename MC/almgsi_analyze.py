import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
import numpy as np
import json
from scipy.spatial import ConvexHull
from scipy.interpolate import griddata
from cemc.tools import CanonicalFreeEnergy
from matplotlib import pyplot as plt
from ase.units import kJ, mol
from scipy.interpolate import griddata
from mpl_toolkits.mplot3d import axes3d, Axes3D
from ase.calculators.cluster_expansion import ClusterExpansion
from ase.ce import BulkCrystal
from ase.ce import CorrFunction
import dataset
plt.switch_backend("TkAgg")

mc_db_name = "data/enthalpy_formation_almgsi.db"
#fig = mlab.figure()

def get_formula(mg_conc,si_conc):
    n_mg = int(100*mg_conc)
    n_si = int(100*si_conc)
    formula = "Al{}Mg{}Si{}".format(100-n_mg-n_si,n_mg,n_si)
    return formula

def get_free_energies():
    db = dataset.connect( "sqlite:///"+mc_db_name )
    tbl = db["systems"]
    result = {}
    res_tbl = db["results"]
    for row in tbl.find(status="finished"):
        formula = get_formula(row["mg_conc"],row["si_conc"])
        internal_energy = []
        temperature = []
        mg_conc = row["mg_conc"]
        si_conc = row["si_conc"]
        conc = {"Mg":mg_conc,"Si":si_conc,"Al":1.0-mg_conc-si_conc}
        n_atoms = 1000
        for resrow in res_tbl.find(sysID=row["id"]):
            internal_energy.append( resrow["internal_energy"]/n_atoms )
            temperature.append( resrow["temperature"] )
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

def get_ref_energies():
    conc_args = {
        "conc_ratio_min_1":[[64,0,0]],
        "conc_ratio_max_1":[[0,64,0]],
        "conc_ratio_min_2":[[64,0,0]],
        "conc_ratio_max_2":[[0,0,64]]
    }
    orig_spin_dict = {
        "Mg":1.0,
        "Si":-1.0,
        "Al":0.0
    }

    kwargs = {
        "crystalstructure":"fcc",
        "a":4.05,
        "size":[4,4,4],
        "basis_elements":[["Mg","Si","Al"]],
        "conc_args":conc_args,
        "db_name":"data/almgsi_excess.db",
        "max_cluster_size":4
    }
    ceBulk = BulkCrystal( **kwargs )
    ceBulk.spin_dict = orig_spin_dict
    ceBulk.basis_functions = ceBulk._get_basis_functions()
    ceBulk._get_cluster_information()
    ceBulk.reconfigure_settings()
    eci_file = "data/almgsi_fcc_eci.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    cf = CorrFunction( ceBulk )
    atoms = ceBulk.atoms
    cluster_names = ecis.keys()
    for atom in atoms:
        atom.symbol = "Al"
    corr_func = cf.get_cf_by_cluster_names(atoms, cluster_names)
    ref_energies = {}
    ref_energies["Al"] = get_energy( corr_func, ecis )
    for atom in atoms:
        atom.symbol = "Mg"
    corr_func = cf.get_cf_by_cluster_names(atoms, cluster_names)
    ref_energies["Mg"] =  get_energy( corr_func, ecis )
    for atom in atoms:
        atom.symbol = "Si"
    corr_func = cf.get_cf_by_cluster_names(atoms, cluster_names)
    ref_energies["Si"] =  get_energy( corr_func, ecis )
    return ref_energies

def get_energy( cf, eci ):
    energy = 0.0
    for key,value in eci.iteritems():
        energy += value*cf[key]
    return energy

def excess():
    ref_energies = {
        "Mg":-1.599,
        "Si":-4.864,
        "Al":-3.737
    }
    ref_energies = get_ref_energies()
    print (ref_energies)
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
    fig_solute = plt.figure()
    ax_solute = fig_solute.add_subplot(1,1,1)
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
        unique_si_concs = unique_si_concs[unique_si_concs<0.9]
        cm = plt.cm.get_cmap('viridis')
        convex_hull3D( mg_concs, si_concs, form_energy, formulas )
        x_sol = si_concs/(mg_concs+si_concs)
        for c_si in unique_si_concs:
            c_mg = mg_concs[si_concs==c_si]
            e_form = form_energy[si_concs==c_si]
            forms = np.array(formulas)[si_concs==c_si]
            srt = np.argsort(c_mg)
            c_mg = [c_mg[indx] for indx in srt]
            e_form = [e_form[indx] for indx in srt]
            print (c_mg)

            scaled_si_conc = (c_si-np.min(unique_si_concs))/(np.max(unique_si_concs)-np.min(unique_si_concs))
            color=cm(scaled_si_conc)
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

        # Solute plot
        im = ax_solute.scatter( x_sol, form_energy, c=si_concs, cmap="copper", marker="v", vmin=0,vmax=0.5 )

    cbar = fig_solute.colorbar(im, orientation="horizontal")
    cbar.set_label("Si concentration")
    ax_solute.set_ylabel("Enthalpy of formation (kJ/mol)")
    ax_solute.set_xlabel( "\$c_\mathrm{Si}/(\c_\mathrm{Si}+c_\mathrm{Mg})\$")
    ax_solute.spines["right"].set_visible(False)
    ax_solute.spines["top"].set_visible(False)
    add_covnex_hull(x_sol[1:], form_energy[1:], ax_solute, formulas=formulas[1:])
    plt.show()

def convex_hull3D( mg_conc, si_conc, E, formulas ):
    hull = ConvexHull( np.vstack((mg_conc,si_conc,E)).T )
    printed = []
    mg_concs = []
    si_concs = []
    eng = []
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
