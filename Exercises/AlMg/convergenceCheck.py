import sqlite3 as sq
from matplotlib import pyplot as plt

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
    #cur.execute( "SELECT energy FROM systems WHERE id IN SELECT id FROM runs WHERE tags='kptsconv'" )
    #kptsEneergies = cur.fetchall()
    #cur.execute( "SELECT kpts FROM runs WHERE tags='kptsconv'" )
    #kpts = cur.fetchall()
    con.close()

    # Create plots
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( hspacing, energies )
    plt.show()

if __name__ == "__main__":
    main()
