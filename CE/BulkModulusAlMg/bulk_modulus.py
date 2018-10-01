import sys
import gpaw as gp
from ase.io import read,write
import numpy as np
import matplotlib as mpl
from ase.optimize.precon import PreconLBFGS,Exp
from ase.constraints import UnitCellFilter
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt

def relax(fname):
    atoms = read(fname)
    calc = gp.GPAW(mode=gp.PW(600),kpts=(4,4,4),xc="PBE")
    atoms.set_calculator(calc)
    uf = UnitCellFilter(atoms,hydrostatic_strain=True)
    relaxer = PreconLBFGS( uf, logfile="logfile.log" )
    relaxer.run( fmax=0.025,smax=0.003 )
    write( fname, atoms )

def compute_bulk(fname):
    atoms = read(fname)
    calc = gp.GPAW(mode=gp.PW(600),kpts=(4,4,4),xc="PBE")

    energies = []
    base_fname = fname.split(".")[0]
    lat_params = np.loadtxt( base_fname+"_latparam.csv", delimiter="," )
    V0 = atoms.get_volume()
    a0 = (V0)**(1.0/3.0)
    for a in lat_params:
        sim_atoms = atoms.copy()
        cell = sim_atoms.get_cell()
        cell *= (a/a0)
        sim_atoms.set_cell(cell,scale_atoms=True)
        sim_atoms.set_calculator(calc)
        energy = sim_atoms.get_potential_energy()
        energies.append(energy/len(sim_atoms))

    out = np.vstack((lat_params,energies)).T
    np.savetxt( base_fname+"_bulk.csv",out,delimiter=",",header="Lattice parameter,energy per atom")

def fit( a, energy ):
    A = np.zeros((len(energy),3))
    A[:,0] = 1.0
    A[:,1] = a
    A[:,2] = a**2
    x,res,rank,s = np.linalg.lstsq( A,energy )
    return x

def plot_all():
    fnames = ["al_bulk.csv","mg_bulk.csv","almg_bulk.csv"]
    labels = ["Al","Mg","AlMg"]
    colors = ["#1b9e77","#d95f02","#7570b3"]
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for i,fname in enumerate(fnames):
        data = np.loadtxt(fname, delimiter=",")
        x = fit(data[:,0],data[:,1])
        a = data[:,0]
        a_fit = a_fit = np.linspace(0.9*np.min(a),1.1*np.max(a),101)
        ax.plot( data[:,0],data[:,1], "o",label=labels[i], mfc="none",color=colors[i])
        ax.plot(a_fit,x[0]+a_fit*x[1]+x[2]*a_fit**2,color=colors[i])
    ax.set_xlabel( "Lattice parameter (\$\SI{}{\\angstrom})\$")
    ax.set_ylabel( "Energy (eV/atom)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.legend( loc="best", frameon=False )
    plt.show()

def bulk_modulus(fname):
    data = np.loadtxt(fname, delimiter=",")
    a = data[:,0]
    energy = data[:,1]
    x = fit(a,energy)

    energy_curvature = 2*x[2]
    print ("Curvature: {} eV/A^2".format(energy_curvature))

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot( a,energy, "o" )
    a_fit = np.linspace(0.9*np.min(a),1.1*np.max(a),101)
    ax.plot( a_fit, x[0] + x[1]*a_fit + x[2]*a_fit**2 )
    plt.show()

if __name__ == "__main__":
    fname = sys.argv[1]
    relax(fname)
    #compute_bulk(fname)
    #bulk_modulus(fname)
    #plot_all()
