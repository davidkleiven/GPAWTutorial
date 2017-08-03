import sqlite3 as sqdb
from matplotlib import pyplot as plt
import numpy as np

def main():
    dbname = "aluminum.db"
    ids="(50,51,52,53,54)"

    con = sqdb.connect( dbname )
    cur = con.cursor()
    cur.execute( "SELECT energy FROM systems WHERE ID IN %s"%(ids) )
    energies = cur.fetchall()
    cur.execute( "SELECT LATTICEPARAM FROM PARAMS WHERE ID In %s"%(ids) )
    lattice = cur.fetchall()
    con.close()

    lattice = np.array(lattice)[:,0]
    energies = np.array(energies)[:,0]

    coeff = np.polyfit( lattice, energies, 2 )
    print ("Lattice constant: %.2f"%(-coeff[1]/(2.0*coeff[0])) )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( lattice, energies )
    plt.show()

if __name__ == "__main__":
    main()
