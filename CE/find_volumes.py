import ase.db
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
from scipy.stats import linregress

def volume(atoms):
    return np.linalg.det(atoms.get_cell())

def mg_conc(atoms):
    counter = 0
    for atom in atoms:
        if ( atom.symbol == "Mg" ):
            counter += 1
    return float(counter)/len(atoms)

def fcc_lattice_parameter_from_volume_primitive_cell(V,n_atoms):
    side_length = (4.0*V/n_atoms)**(1.0/3.0)
    a = side_length
    return a

def main():
    db = ase.db.connect("ce_hydrostatic.db")
    volumes = []
    concs = []
    for row in db.select(converged=1):
        atoms = db.get_atoms(id=row.id)
        volumes.append(volume(atoms))
        concs.append(mg_conc(atoms))
    lattice_params = [fcc_lattice_parameter_from_volume_primitive_cell(V,64) for V in volumes]

    fname = "almg_lattice_parameter.csv" # From J. L. Murray, The Al-Mg system, 1982
    data = np.loadtxt( fname, delimiter=",")

    mg_conc_exp = data[:,0]
    lattice_param_exp = data[:,1]*10
    slope, interscept, r_value, p_value, stderr = linregress( concs, lattice_params )
    print (slope,interscept)
    x = np.linspace(0.0,0.6,10)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,interscept+slope*x)
    ax.plot( concs, lattice_params, 'o', label="DFT", mfc="none" )
    ax.plot( mg_conc_exp, lattice_param_exp, 'x', label="Exp")
    ax.legend(loc="best", labelspacing=0.05, frameon=False)
    ax.set_xlabel("Mg concentration")
    ax.set_ylabel("FCC lattice parameter")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
