from __future__ import print_function
import sys
import ase
import gpaw as gp

def originalCalculation():
    d = 1.1
    a = 5.0
    c = a/2.0

    atoms = ase.Atoms( "CO", positions=([c-d/2.0,c,c],[c+d/2.0,c,c]), cell=(a,a,a) )

    calc = gp.GPAW( nbands=5, h=0.2, txt=None )
    atoms.set_calculator(calc)
    energy = atoms.get_potential_energy()
    calc.write( "CO.gpw", mode="all" )

def createCubeFiles():
    atoms, calc = gp.restart( "CO.gpw" )
    nbands = calc.get_number_of_bands()
    for band in range(nbands):
        wf = calc.get_pseudo_wave_function(band=band)
        fname = "wavefuncttionCO_%d.cube"%(band)
        ase.io.write( fname, atoms, data=wf )

def main( argv ):
    if ( argv[0] == "run" ):
        originalCalculation()
    elif ( argv[0] == "export" ):
        createCubeFiles()

if __name__ == "__main__":
    main( sys.argv[1:] )
