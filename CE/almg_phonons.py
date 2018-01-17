import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
sys.path.append("/home/davidkl/GPAWTutorial/CE")
import gpaw as gp
from ase.build import bulk
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
from ase.visualize import view

db_name = "almg_phonons.db"
#db_name = "/home/davidkl/GPAWTutorial/CE/almg_phonons.db"
def main():
    conc_args = {
        "conc_ratio_min_1":[[0,1]],
        "conc_ratio_max_1":[[1,0]],
    }
    cebulk = BulkCrystal( "fcc", 4.05, [2,2,2], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=2, reconf_db=False)
    generator = GenerateStructures(cebulk,struct_per_gen=1)
    generator.generate_probe_structure( num_steps=1 )

def phonon_run(runID):
    db = connect(db_name)
    atoms = db.get_atoms(id=runID)
    #atoms = atoms*(2,2,2)
    calc = gp.GPAW( mode=gp.PW(600), xc="PBE", kpts=(4,4,4), nbands="120%", symmetry="off" )
    #atoms.set_calculator(calc)
    ph = Phonons( atoms, calc, supercell=(1,1,1) )
    #ph.run()
    #return
    ph.read( acoustic=True )
    omega_e, dos_e = ph.dos( kpts=(50,50,50), npts=700, delta=5E-4 )

    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot( omega_e*1000.0, dos_e )
    #ax.set_xlabel( "Energy (meV)" )
    #plt.show()

if __name__ == "__main__":
    runID = int(sys.argv[1])
    phonon_run(runID)
