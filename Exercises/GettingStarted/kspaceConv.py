import sqlite3 as sqdb
from matplotlib import pyplot as plt

def main():
    db = "aluminum.db"
    con = sqdb.connect( db )
    cur = con.cursor()
    cur.execute( "SELECT ID,KPTS FROM PARAMS WHERE REASON LIKE '%Kpt%'")
    selection = cur.fetchall()
    ids = [entry[0] for entry in selection]
    kpts = [entry[1] for entry in selection]
    ids = ",".join(map(str,ids))
    cur.execute( "SELECT energy FROM systems WHERE ID in (%s)"%ids )
    energies = cur.fetchall()
    con.close()

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(kpts,energies)
    plt.show()

if __name__ == "__main__":
    main()
