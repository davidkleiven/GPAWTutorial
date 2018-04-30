from atomtools.ase import ConstituentStrain
import gpaw as gp
from ase.build import bulk
import numpy as np
from matplotlib import pyplot as plt

db_name = "constituent_strain_fcc_almg.db"
def get_calc():
    calc = gp.GPAW( mode=gp.PW(500), xc="PBE", kpts=(8,8,8), nbands=-20, symmetry={'do_not_symmetrize_the_density': True} )
    return calc

def full_relax():
    al = bulk("Al",a=4.05, crystalstructure="fcc" )
    mg = bulk("Mg", a=4.05, crystalstructure="fcc" )
    calc_al = get_calc()
    calc_mg = get_calc()
    al.set_calculator( calc_al )
    mg.set_calculator( calc_mg )
    strain_al = ConstituentStrain( atoms=al, db_name=db_name )
    strain_mg = ConstituentStrain( atoms=mg, db_name=db_name )
    strain_al.full_relaxation()
    strain_mg.full_relaxation()

def relax():
    direction = (0,1,1)
    ax = np.linspace(3.3,5.0,10)
    al = bulk("Al",a=4.05, crystalstructure="fcc" )
    mg = bulk("Mg", a=4.6, crystalstructure="fcc" )
    calc_al = get_calc()
    calc_mg = get_calc()
    al.set_calculator( calc_al )
    mg.set_calculator( calc_mg )
    strain_al = ConstituentStrain( atoms=al, db_name=db_name )
    strain_mg = ConstituentStrain( atoms=mg, db_name=db_name )
    #strain_al.run( cell_length_x=ax, direction=direction )
    strain_mg.run( cell_length_x=ax, direction=direction )

def plot():
    direction = (0,1,1)
    al = bulk("Al",a=4.05, crystalstructure="fcc" )
    strain_al = ConstituentStrain( atoms=al, db_name=db_name )
    strain_al.plot_strain_energies( direction=direction )

    fig = plt.figure()
    dirs = [(0,0,1),(0,1,1)]
    ax = fig.add_subplot(1,1,1)
    for direction in dirs:
        conc, E_CS = strain_al.coherency_strain_energy( direction=direction )
        ax.plot( conc, E_CS )
    plt.show()
if __name__ == "__main__":
    #full_relax()
    relax()
    #plot()
