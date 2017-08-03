from __future__ import print_function
from ase import Atoms
from ase.visualize import view
import gpaw as gp
import sqlite3 as sqdb
import datetime
from ase import db

def main():
    database = "aluminum.db"
    # Parse parameters from the database
    con = sqdb.connect( database )
    cur = con.cursor()
    cur.execute( "SELECT VIEW,CUTOFF,KPTS,LATTICEPARAM,_rowid_,STRUCTURE FROM PARAMS WHERE STATUS=?", ("RUN",) )
    jobs = cur.fetchall()
    con.close()
    print (jobs)

    for job in jobs:
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        structure = job[5]
        a = job[3]
        show = job[0]
        b = a/2.0

        if ( structure == "FCC" ):
            bulk = Atoms( "Al", cell=[[0,b,b],
                                      [b,0,b],
                                      [b,b,0]], pbc=True)
        elif ( structure == "BCC" ):
            bulk = Atoms( "Al", cell=[[b,b,b],
                                      [b,b,-b],
                                      [b,-b,b]], pbc=True )
        else:
            print ( "Unknown lattice type" )
            continue

        if ( show == 1 ):
            view( bulk )

        calcfile = "data/alum"+structure+stamp+".txt"
        cutoff = job[1]
        k = job[2]
        calc = gp.GPAW( mode=gp.PW(cutoff), kpts=(k,k,k), txt=calcfile, xc="LDA" )

        bulk.set_calculator( calc )
        energy = bulk.get_potential_energy()

        gpwfile = "data/alum"+structure+stamp+".gpw"
        calc.write( gpwfile )

        # Update the database
        aseDB = db.connect( database )
        lastID = aseDB.write( bulk )

        con = sqdb.connect( database )
        cur = con.cursor()
        row = int(job[4])
        cur.execute( "UPDATE PARAMS SET GPWFILE=?,TXTFILE=?,STATUS=?,ID=? WHERE _rowid_=?", (gpwfile,calcfile,"FINISHED",lastID,row) )
        con.commit()
        con.close()

if __name__ == "__main__":
    main()
