import sqlite3 as sq
from matplotlib import pyplot as plt
import numpy as np

def main():
    db_name = "AlMg.db"

    con = sq.connect( db_name )
    cur = con.cursor()

    # Extract hspace convergence
    cur.execute( "SELECT hspacing,resultID FROM runs WHERE tags='convergence'" )
    res = cur.fetchall()
    hspacing = [entry[0] for entry in res]
    resultID = [entry[1] for entry in res]
    energies = []

    for resID in resultID:
        cur.execute( "SELECT energy FROM systems WHERE id=?", (resID,) )
        energies.append( cur.fetchall()[0] )

    # Extract k-sampling convergence
    cur.execute( "SELECT kpts,resultID FROM runs WHERE tags='kptsConv2'" )
    res = cur.fetchall()
    kpts = [entry[0] for entry in res]
    resID = [entry[1] for entry in res]
    kptsEnergies = []
    for rid in resID:
        cur.execute( "SELECT energy FROM systems WHERE ID=?", (rid,) )
        kptsEnergies.append( cur.fetchall()[0] )

    # Extract band convergence
    cur.execute( "SELECT nbands,resultID FROM runs WHERE tags='bandconv'" )
    res = cur.fetchall()
    nbands,resultID = zip( *res )
    bandeng = []
    for rid in resultID:
        cur.execute( "SELECT energy FROM systems WHERE ID=?", (rid,) )
        bandeng.append( cur.fetchall()[0] )
    con.close()

    # Create plots
    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    ax.plot( hspacing, energies )
    ax2 = fig.add_subplot(1,3,2)
    ax2.plot( kpts, kptsEnergies )

    ax3 = fig.add_subplot(1,3,3)
    nbands = np.array( nbands )
    ax3.plot( np.abs(nbands), bandeng )
    plt.show()

if __name__ == "__main__":
    main()
