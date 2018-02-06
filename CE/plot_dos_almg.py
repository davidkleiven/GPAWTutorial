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
    res = res[0]
    db = connect(db_name)
    atoms = db.get_atoms(id=res["atID"])
    dw = res["omega_e"][1]-res["omega_e"][0]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( res["omega_e"], res["dos_e"] )
    plt.show()

if __name__ == "__main__":
    main()
