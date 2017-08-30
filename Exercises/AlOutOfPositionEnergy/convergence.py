import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
import sqlite3 as sq

def main():
    db_name = "aloutofpos.db"
    cutoffConv = {
        "cutoff":[],
        "systemID":[],
        "energy":[]
    }

    kptConv = {
        "kpts":[],
        "systemID":[],
        "energy":[]
    }

    kptConvGamma = {
        "kpts":[],
        "systemID":[],
        "energy":[]
    }

    bandConv = {
        "bands":[],
        "systemID":[],
        "energy":[]
    }
    con = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "SELECT cutoff,systemID FROM simpar WHERE ID IN (2,3,4,5,6)")
    params = cur.fetchall()
    cutoffConv["cutoff"] = [entry[0] for entry in params]
    cutoffConv["systemID"] = [entry[1] for entry in params]

    cur.execute( "SELECT kpts,systemID FROM simpar WHERE ID IN (7,8,9,10,18)")
    params= cur.fetchall()
    kptConv["kpts"] = [entry[0] for entry in params]
    kptConv["systemID"] = [entry[1] for entry in params]

    cur.execute( "SELECT kpts,systemID FROM simpar WHERE ID IN (19,20,21,22)")
    params= cur.fetchall()
    kptConvGamma["kpts"] = [entry[0] for entry in params]
    kptConvGamma["systemID"] = [entry[1] for entry in params]

    cur.execute( "SELECT nbands,systemID FROM simpar WHERE ID IN (11,12,13,14,15,16)")
    params= cur.fetchall()
    bandConv["bands"] = [entry[0] for entry in params]
    bandConv["systemID"] = [entry[1] for entry in params]

    # Extract the relevant data from the system table
    for systID in cutoffConv["systemID"]:
        cur.execute( "SELECT energy FROM systems WHERE ID=?", (systID,) )
        cutoffConv["energy"].append( cur.fetchone()[0] )

    for systID in kptConv["systemID"]:
        cur.execute( "SELECT energy FROM systems WHERE ID=?", (systID,) )
        kptConv["energy"].append( cur.fetchone()[0] )

    for systID in kptConvGamma["systemID"]:
        cur.execute( "SELECT energy FROM systems WHERE ID=?", (systID,) )
        kptConvGamma["energy"].append( cur.fetchone()[0] )

    for systID in bandConv["systemID"]:
        cur.execute( "SELECT energy FROM systems WHERE ID=?", (systID,) )
        bandConv["energy"].append( cur.fetchone()[0] )

    # Create figures
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot( cutoffConv["cutoff"], cutoffConv["energy"] )
    ax1.set_xlabel("Plane Wave Cutoff Energy (eV)" )
    ax1.set_ylabel("Energy (eV/atom)")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    kpt = np.argsort(kptConv["kpts"])
    ax2.plot( np.array(kptConv["kpts"])[kpt], np.array(kptConv["energy"])[kpt], label="Regular" )
    ax2.plot( kptConvGamma["kpts"], kptConvGamma["energy"], label="Shifted")
    ax2.set_xlabel("Number of k-points" )
    ax2.set_ylabel("Energy (eV/atom)")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.legend( loc="best", frameon=False )

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(1,1,1)
    ax3.plot( -np.array(bandConv["bands"]), bandConv["energy"] )
    ax3.set_xlabel("Number of extra bands" )
    ax3.set_ylabel("Energy (eV/atom)")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
