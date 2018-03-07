import matplotlib as mpl
mpl.rcParams["font.size"] = 18
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
from ase.db import connect
import numpy as np

db_name = "ce_hydrostatic_eam_relax_effect_atoms.db"

def main():
    db = connect( db_name )
    init_energy = []
    relaxed_energy = []
    for row in db.select( [("converged","=",1)] ):
        relaxed_energy.append( row.energy/row.natoms )
        init_energy.append( row.init_energy/row.natoms )

    init_energy = np.array( init_energy )
    relax_energy = np.array( relaxed_energy)
    diff = relax_energy-init_energy
    hist, bins = np.histogram(diff*1000.0, bins="auto" )
    print (bins)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.bar( bins[1:], hist, 0.4 )
    ax.set_xlabel( "Relaxation energy (meV/atom)" )
    ax.set_ylabel( "Number of configurations" )
    ax.spines["right"].set_visible( False )
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
