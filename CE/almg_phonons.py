import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")
sys.path.append("/home/davidkl/GPAWTutorial/CE")
import gpaw as gp
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.calculators.eam import EAM
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
from nnpotential.nn_calculator import NN
from nnpotential.neural_network import NNPotential
from ase.units import kB
import numpy as np
from atomtools.ce import ce_phonon_dos as cpd
from ase.dft.kpoints import ibz_points, bandpath

wrk = "/home/davidkl/Documents/GPAWTutorial/CE"
db_name = wrk+"/"+"ce_hydrostatic.db"
#db_name = "/home/davidkl/GPAWTutorial/CE/almg_phonons.db"

def aucu_phonons():
    N = 7
    atoms = bulk("Au", crystalstructure="fcc", a=4.08 )
    calc = EMT()
    atoms.set_calculator(calc)

    ph = Phonons( atoms, calc, supercell=(N,N,N), delta=0.05 )
    ph.run()
    ph.read( acoustic=True )
    ph.clean()
    omega_e_au, dos_e_au = ph.dos( kpts=(50,50,50), npts=1000, delta=5E-4 )

    atoms = bulk("Cu", crystalstructure="fcc", a=3.62 )
    atoms.set_calculator(calc)

    ph = Phonons( atoms, calc, supercell=(N,N,N), delta=0.05 )
    ph.run()
    ph.read( acoustic=True )
    ph.clean()
    omega_e_cu, dos_e_cu = ph.dos( kpts=(13,13,13), npts=100, delta=5E-4 )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( omega_e_au*1000.0, dos_e_au )
    ax.plot( omega_e_cu*1000.0, dos_e_cu )
    ax.set_xlabel( "Energy (meV)" )

    logw_au = np.sum(np.log(omega_e_au[1:])*dos_e_au[1:])
    logw_cu = np.sum(np.log(omega_e_cu[1:])*dos_e_cu[1:])
    print (logw_au,logw_cu,logw_au-logw_cu)
    plt.show()

def main():
    conc_args = {
        "conc_ratio_min_1":[[0,1]],
        "conc_ratio_max_1":[[1,0]],
    }
    cebulk = BulkCrystal( "fcc", 4.05, [2,2,2], 1, [["Al","Mg"]], conc_args, db_name, max_cluster_size=2, reconf_db=False)
    generator = GenerateStructures(cebulk,struct_per_gen=1)
    generator.generate_probe_structure( num_steps=1 )

def phonon_run(runID, save_to_db=False, plot_bands=False ):
    print ("Running ID %d"%(runID))
    db = connect(db_name)
    atoms = db.get_atoms(id=runID)
    #view(atoms)
    #atoms = bulk("Al")
    #atoms = atoms*(2,1,1)
    #calc = EAM(potential="/home/davidkl/Documents/EAM/Al-LEA.eam.alloy")
    calc = EAM(potential="/home/davidkl/Documents/EAM/mg-al-set.eam.alloy")
    atoms.set_calculator(calc)
    #calc = gp.GPAW( mode=gp.PW(600), xc="PBE", kpts=(4,4,4), nbands="120%", symmetry="off" )
    #atoms.set_calculator(calc)
    ph = Phonons( atoms, calc, supercell=(3,3,3), name=wrk+"/phonon_files/phonon%d"%(runID) )
    ph.run()
    #return
    ph.read( acoustic=True )
    omega_e, dos_e = ph.dos( kpts=(30,30,30), npts=1000, delta=5E-4 )
    if ( plot_bands ):
        points = ibz_points['fcc']
        G = points['Gamma']
        X = points['X']
        W = points['W']
        K = points['K']
        L = points['L']
        U = points['U']
        point_names = ['$\Gamma$', 'X', 'U', 'L', '$\Gamma$', 'K']
        path = [G, X, U, L, G, K]

        path_kc, q, Q = bandpath(path, atoms.cell, 100)
        omega_kn = 1000.0*ph.band_structure( path_kc )

        figb = plt.figure()
        axb = figb.add_subplot(1,1,1)
        for n in range(len(omega_kn[0])):
            omega_n = omega_kn[:,n]
            axb.plot( q, omega_n )
        plt.show()

    if ( save_to_db ):
        # Store the results in the database
        db.update( runID, has_dos=True )

        manager = cpd.PhononDOS_DB(db_name)

        # Extract relevant information from the atoms database
        row = db.get(id=runID)
        name = row.name
        atID = row.id
        manager.save( name=name, atID=atID, omega_e=omega_e, dos_e=dos_e )

    #fig = plt.figure()
    #ax = fig.add_subplot(1,1,1)
    #ax.plot( omega_e*1000.0, dos_e )
    #ax.set_xlabel( "Energy (meV)" )
    #plt.show()
    #ph.clean()

if __name__ == "__main__":
    runID = int(sys.argv[1])
    phonon_run(runID,plot_bands=False)
    #aucu_phonons()
