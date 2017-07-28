from __future__ import print_function
from ase import Atoms
from ase import Atom
from ase.calculators.emt import EMT
from ase.optimize import QuasiNewton
import gpaw as gp
from ase import db
import sqlite3 as sq

def emtOptimize():
    system = Atoms( "H2", positions=[[0.0,0.0,0.0],
    [0.0,0.0,1.0]])

    calc = EMT()

    system.set_calculator( calc )
    opt = QuasiNewton( system, trajectory="h2.emt.traj" )
    opt.run( fmax=0.05 )

def optimizeH2O( dbname ):
    database = db.connect( dbname )
    sqdb = sq.connect( dbname )
    cur = sqdb.cursor()
    cur.execute( "SELECT _rowid_,CellSize,XC,JobType,AtomID FROM InputParams WHERE STATUS=?", ("RUN",) )
    jobs = cur.fetchall()
    sqdb.close()

    for job in jobs:
        jobtype = job[3]
        atomID = int( job[4] )
        if ( atomID >= 0 ):
            print ( "Using Atoms object with ID %d"%(atomID) )
            system = database.get_atoms( selection=atomID )
        else:
            system = Atoms( ["H","H","O"], positions=[[3.0,3.8,2.4],[3,2.23,2.4],[3,3,3]] )

        size = job[1]
        system.set_cell([size,size,size])
        system.center()

        xc = job[2]
        calc = gp.GPAW( xc=xc )

        system.set_calculator( calc )

        if ( jobtype == "Optimize" ):
            opt = QuasiNewton( system, trajectory="H2O.gpw.traj" )
            opt.run( fmax=0.05 )
        elif ( jobtype == "GS" ):
            e1 = system.get_potential_energy()
        lastID = database.write( system )
        row = int(job[0])

        # Update ID
        print ("Updating entry in database")
        sqdb = sq.connect( dbname )
        cur = sqdb.cursor()
        cur.execute( "UPDATE InputParams SET ID=?, STATUS=? WHERE _rowid_=?",(lastID,"FINISHED",row) )
        sqdb.commit()
        sqdb.close()
        print ("Database updated")

def atomizationEnergyH2O( dbname, xc ):
    database = db.connect( dbname )
    system = database.get_atoms( selection=20 ) # Use atom with reference 20 as structure

    calc = gp.GPAW( xc=xc )
    system.set_calculator( calc )
    eH2O = system.get_potential_energy()

    calc = gp.GPAW( xc=xc, hund=True )
    Hsyst = Atoms( "H", positions=[[3.5,3.5+0.001,3.5+0.0002]] )
    Hsyst.set_cell([7,7,7])
    Hsyst.center()
    Hsyst.set_calculator( calc )
    eH = Hsyst.get_potential_energy()

    Osyst = Atoms( "O", positions=[[3.5,3.5+0.001,3.5+0.0002]] )
    Osyst.set_cell([7,7,7])
    Osyst.center()
    Osyst.set_calculator( calc )
    eO = Osyst.get_potential_energy()

    print ("Hydrogen energy: %.2E"%(eH))
    print ("Oxygen energy: %.2E"%(eO))
    print ("Atomization energy: %.2E"%(eH2O - 2*eH - eO))

def main():
    #emtOptimize()
    #optimizeH2O( "h2O.db" )
    atomizationEnergyH2O( "h2O.db", "PBE")

if __name__ == "__main__":
    main()
