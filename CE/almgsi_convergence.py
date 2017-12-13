import gpaw as gp
import ase.db
from ase.build import bulk
import random
import os
import sys
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
import numpy as np

def db_name():
    candidates = ["almgsi_conv.db","/home/davidkl/GPAWTutorial/CE/almgsi_conv.db"]
    for cand in candidates:
        if ( os.path.exists(cand) ):
            return cand
    return candidates[0]

def prepare_db():
    atoms = bulk("Al", a=4.15)
    atoms = atoms*(4,4,4)
    for i in range(0,42):
        if ( i < 21 ):
            atoms[i].symbol = "Si"
        else:
            atoms[i].symbol = "Mg"

    symbols = [atom.symbol for atom in atoms]
    random.shuffle(symbols)
    for symb,atom in zip(symbols,atoms):
        atom.symbol = symb

    db = ase.db.connect( db_name() )
    for i in range(8):
        db.write(atoms)

def analyze():
    db = ase.db.connect( db_name() )
    kpts = []
    cutoff = []
    energy = []
    for row in db.select():
        try:
            new_kpt = row.n_kpt
            new_cut = row.cutoff
            new_eng = row.trial_energy
            kpts.append( new_kpt )
            cutoff.append( new_cut )
            energy.append( new_eng )
        except:
            pass

    kpt_kpt = []
    eng_kpt = []
    for i in range(len(energy)):
        if ( cutoff[i] == 500 ):
            kpt_kpt.append( kpts[i] )
            eng_kpt.append( energy[i] )

    srt_indx = np.argsort(kpt_kpt)
    kpt_kpt = [kpt_kpt[indx] for indx in srt_indx]
    eng_kpt = [eng_kpt[indx] for indx in srt_indx]

    cut_cut = []
    eng_cut = []
    for i in range(len(energy)):
        if ( kpts[i] == 1 ):
            cut_cut.append(cutoff[i])
            eng_cut.append(energy[i])
    srt_indx = np.argsort(cut_cut)
    cut_cut = [cut_cut[indx] for indx in srt_indx]
    eng_cut = [eng_cut[indx] for indx in srt_indx]

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1,1,1)
    ax1.plot( kpt_kpt, eng_kpt, "-o" )
    ax1.set_xlabel( "Number of k-points" )
    ax1.set_ylabel( "Energy (eV)" )

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(1,1,1)
    ax2.plot( cut_cut, eng_cut, "-o" )
    ax2.set_xlabel( "Plane wave cutoff (eV)" )
    ax2.set_ylabel( "Energy (eV)")
    plt.show()

def main( runID ):
    db = ase.db.connect( db_name() )
    atoms = db.get_atoms( id=runID )
    row = db.get( id=runID )
    n_kpt = row.n_kpt
    cutoff = row.cutoff

    calc = gp.GPAW( mode=gp.PW(cutoff), xc="PBE", kpts=(n_kpt,n_kpt,n_kpt), nbands="120%" )
    atoms.set_calculator( calc )
    energy = atoms.get_potential_energy()
    db.update( runID, trial_energy=energy )

if __name__ == "__main__":
    arg = sys.argv[1]
    if ( arg == "init" ):
        prepare_db()
    elif ( arg == "analyze" ):
        analyze()
    else:
        # Arg should be a run ID
        main(arg)
