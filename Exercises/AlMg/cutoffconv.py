import sqlite3 as sq
from matplotlib import pyplot as plt

def main():
    db_name = "AlMg.db"

    con = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "SELECT cutoff,resultID FROM runs WHERE tags='cutoffconv'")
    cutoff,resultID = zip( *cur.fetchall() )

    energy = []
    for resID in resultID:
        cur.execute( "SELECT energy FROM systems WHERE ID=?", (resID,) )
        energy.append( cur.fetchall()[0] )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( cutoff,energy )
    plt.show()

if __name__ == "__main__":
    main()
