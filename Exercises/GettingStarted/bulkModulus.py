import sqlite3 as sqdb
from matplotlib import pyplot as plt
import numpy as np

def main():
    ids = [60,61,62,63,64]

    con = sqdb.connect( "aluminum.db" )
    cur = con.cursor()
    ids = ",".join(map(str,ids))
    cur.execute( "SELECT LATTICEPARAM FROM PARAMS WHERE ID in (%s)"%(ids) )
    latticeParam = cur.fetchall()

    cur.execute( "SELECT energy FROM systems WHERE ID in (%s)"%(ids) )
    energies = cur.fetchall()
    con.close()

    latticeParam = np.array( latticeParam )[:,0]
    energies = np.array( energies )[:,0]
    print (latticeParam)

    # Fit a parabola to the datapoints
    coeff = np.polyfit( latticeParam, energies, 2 )

    a = np.linspace( 0.8*np.min(latticeParam), 1.2*np.max(latticeParam), 100 )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( latticeParam, energies, "ko" )
    ax.plot( a, coeff[0]*a**2 + coeff[1]*a + coeff[2] )
    plt.show()

    eqLattice = 4.05
    bulkmod = 2.0*4.0/(9.0*eqLattice)*coeff[0]*160
    print ("Bulk modulys %.2E GPa"%(bulkmod))

if __name__ == "__main__":
    main()
