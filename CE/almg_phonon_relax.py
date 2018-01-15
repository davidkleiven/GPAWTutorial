import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
sys.path.append("/home/davidkl/GPAWTutorial/CE")
from ase.build import bulk
import gpaw as gp
from matplotlib import pyplot as plt
from ase.db import connect
from ase.optimize.precon import PreconLBFGS,Exp
from ase.io.trajectory import Trajectory
from save_to_db import SaveToDB
import sqlite3 as sq
from ase.constraints import UnitCellFilter

#db_name = "almg_phonons.db"
db_name = "/home/davidkl/GPAWTutorial/CE/almg_phonons.db"

def relax(runID):
    db = connect(db_name)
    atoms = db.get_atoms(id=runID)

    con = sq.connect( db_name )
    cur = con.cursor()
    cur.execute( "SELECT value FROM text_key_values WHERE id=? AND key='name'", (runID,) )
    name = cur.fetchone()[0]
    con.close()

    calc = gp.GPAW( mode=gp.PW(600), xc="PBE", kpts=(4,4,4), nbands="120%", symmetry="off" )
    atoms.set_calculator(calc)
    precon = Exp(mu=1.0,mu_c=1.0)
    save_to_db = SaveToDB(db_name,runID,name)
    logfile = "logfile%d.log"%(runID)
    traj = "trajectory%d.traj"%(runID)
    uf = UnitCellFilter(atoms, hydrostatic_strain=True )
    relaxer = PreconLBFGS( uf, logfile=logfile, use_armijo=True, precon=precon )
    relaxer.attach(save_to_db, interval=1, atoms=atoms)
    trajObj = Trajectory(traj,"w",atoms)
    relaxer.attach(trajObj)
    relaxer.run( fmax=0.05, smax=0.003 )

if __name__ == "__main__":
    runID = int(sys.argv[1])
    relax(runID)
