import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
from ase.build import bulk
import gpaw as gp
from ase.calculators.emt import EMT
from ase.phonons import Phonons
from matplotlib import pyplot as plt
from ase.ce.settings import BulkCrystal
from ase.ce.newStruct import GenerateStructures
from ase.db import connect
from ase.optimize.precon import PreconLBFGS,Exp
from ase.io.trajectory import Trajectory
from save_to_db import SaveToDB
import sqlite3 as sq
from ase.constraints import UnitCellFilter

#db_name = "almg_phonons.db"
db_name = "/home/davidkl/GPAWTutorial/CE/almg_phonons.db"
def main():
    conc_args = {
        "conc_ratio_min_1":[[0,1]],
        "conc_ratio_max_1":[[1,0]],
    }
    cebulk = BulkCrystal( "fcc", 4.05, [2,2,2], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=2, reconf_db=False)
    generator = GenerateStructures(cebulk,struct_per_gen=1)
    generator.generate_probe_structure( num_steps=1 )

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

def phonon_run(runID):
    atoms = atoms*(2,2,2)
    calc = gp.GPAW( mode=gp.PW(600), xc="PBE", kpts=(4,4,4), nbands="120%", symmetry="off" )
    #calc = EMT()
    ph = Phonons( atoms, calc, supercell=(2,2,2) )
    ph.run()
    #return
    ph.read( acoustic=True )
    omega_e, dos_e = ph.dos( kpts=(50,50,50), npts=700, delta=5E-4 )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( omega_e*1000.0, dos_e )
    ax.set_xlabel( "Energy (meV)" )
    plt.show()

if __name__ == "__main__":
    runID = int(sys.argv[1])
    relax(runID)
    #main()
    #phonon_run(bulk("Al"))
