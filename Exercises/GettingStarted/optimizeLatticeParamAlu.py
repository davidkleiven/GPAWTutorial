from __future__ import print_function
from ase import Atoms
from ase import database
import sqlite3 as sqdb
from ase.visualize import view

def main():
    database = "aluminum.db"

    con = sqdb.connect( database )
    cur = con.cursor()
    cur.execute( "SELECT VIEW,_rowid_,STRUCTURE FROM OPTIMIMIZESTRUCTURE WHERE STATUS=?", ("RUN",) )
    jobs = cur.fetchall()
    con.close()

    for job in jobs:
        
