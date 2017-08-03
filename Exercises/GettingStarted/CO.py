from ase import Atoms
from gpaw import GPAW
from ase.build import molecule
from ase.io.trajectory import Trajectory
from ase.optimize import QuasiNewton
import os
import h5py as h5

def main():
    CO = molecule("CO")
    fname = "data/CO.traj"

    calc = GPAW(
        h=0.2, txt="data/CO.txt"
    )

    CO.set_cell([6,6,6])
    CO.center()

    if ( os.path.isfile(fname) ):
        traj = Trajectory(fname)
        CO = traj[-1]
    else:
        # Optimize the CO structure
        opt = QuasiNewton( CO, trajectory=fname )
        opt.run( fmax=0.05 )

    fname = "data/CO.gpw"

    CO.set_calculator( calc )
    CO.get_potential_energy()
    calc.write( fname )

    for band in range( calc.get_number_of_bands() ):
        wf = calc.get_pseudo_wave_function( band=band )
        with h5.File("data/CO_%d.cmap"%(band) ) as hf:
            hf.create_dataset("wf",data=wf)

if __name__ == "__main__":
    main()
