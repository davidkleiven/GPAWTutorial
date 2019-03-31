import sqlite3 as sq
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
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
    ax.set_xlabel( "Plane wave cut-off energy (eV)" )
    ax.set_ylabel( "Energy (ev/atom)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
