from __future__ import print_function
import sys
import ase
import gpaw as gp
from ase.build import bulk
from ase.visualize import view
import numpy as np


def originalCalculation():
    atoms = bulk("Al", cubic=True)

    calc = gp.GPAW(mode=gp.PW(350), nbands=-50, xc="PBE", kpts={"density": 1.37, "even": True})
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    calc.write( "al.gpw", mode="all" )
    n = calc.get_all_electron_density(gridrefinement=4)
    np.save("density_al.npy", n)
    view(atoms, viewer="avogadro", data=n)
    ase.io.write("al_density.cube", atoms, data=n)


def visualizations():
    atoms = bulk("Al", cubic=True)
    atoms[1].symbol = "Mg"
    atoms[2].symbol = "Mg"
    atoms[3].symbol = "Zn"
    view(atoms, viewer="avogadro")


def show_density(fname):
    from mayavi import mlab
    data = np.load(fname)
    # Display the electron density distribution
    source = mlab.pipeline.scalar_field(data)
    min = data.min()
    max = data.max()
    vol = mlab.pipeline.volume(source, vmin=min + 0.5 * (max - min),
                        vmax=min + 0.6 * (max - min))

    # Add legend to plot
    vol.lut_manager.show_scalar_bar = True
    vol.lut_manager.scalar_bar.orientation = 'vertical'
    vol.lut_manager.scalar_bar.width = 0.001
    vol.lut_manager.scalar_bar.height = 0.04
    vol.lut_manager.scalar_bar.position = (0.01, 0.15)
    vol.lut_manager.number_of_labels = 5
    vol.lut_manager.data_name = "ED"
    mlab.show()


def createCubeFiles():
    atoms, calc = gp.restart( "CO.gpw" )
    nbands = calc.get_number_of_bands()
    for band in range(nbands):
        wf = calc.get_pseudo_wave_function(band=band)
        fname = "wavefuncttionCO_%d.cube"%(band)
        ase.io.write( fname, atoms, data=wf )


def main(argv):
    if argv[0] == "run":
        originalCalculation()
    elif argv[0] == "export":
        createCubeFiles()
    elif argv[0] == "show":
        show_density(argv[1])
    elif argv[0] == "vis":
        visualizations()

if __name__ == "__main__":
    main( sys.argv[1:] )
