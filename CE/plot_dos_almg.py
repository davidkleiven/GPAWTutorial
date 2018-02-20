from atomtools.ce import ce_phonon_dos as cpd
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt
from ase.db import connect
import numpy as np
from ase.units import kB

db_name = "ce_hydrostatic_phonons.db"

def main():
    manager = cpd.PhononDOS_DB(db_name)
    res = manager.get_all()
    db = connect(db_name)
    for i in range(len(res)):
        row = db.get(id=res[i]["atID"])
        if ( not row.converged ):
            print (i,res[i]["atID"])
    db = connect(db_name)
    special_structures = ["Al64","Mg64","Al32Mg32"]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    special_dos = []
    light_scheme = ['#8dd3c7','#ffffb3','#bebada','#fb8072','#80b1d3','#fdb462','#b3de69','#fccde5','#d9d9d9','#bc80bd','#ccebc5','#ffed6f']
    cur_col = 0
    V = None
    for dos in res:
        print (dos["atID"])
        row = db.get( id=dos["atID"] )
        formula = row.formula
        if ( V is None ):
            V = row.volume
        if ( formula in special_structures ):
            dos["formula"] = formula
            special_dos.append(dos)
        else:
            ax.plot( dos["omega_e"]*1000.0, dos["dos_e"]/V, color=light_scheme[cur_col] )
            cur_col += 1
            cur_col = cur_col%len(light_scheme)

    # Plot the special structures
    for dos in special_dos:
        ax.plot( dos["omega_e"]*1000.0, dos["dos_e"]/V, lw=4, label=dos["formula"] )
    ax.legend( loc="best", frameon=False )
    ax.set_xlabel( "Energy (meV)" )
    ax.set_ylabel( "VDOS (\$\SI{}{\milli\electronvolt^{-1}\\angstrom^{-3}}\$")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
