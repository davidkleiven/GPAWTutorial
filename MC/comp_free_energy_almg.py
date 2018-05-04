
import sys
from cemc.mcmc import sgc_montecarlo as sgc
import json
from ase.ce import BulkCrystal
from cemc.wanglandau.ce_calculator import CE, get_ce_calc
from cemc.mcmc import SGCCompositionFreeEnergy, FreeEnergyMuTempArray
from matplotlib import pyplot as plt
import numpy as np

run_array = True
def main():
    conc_args = {
                  "conc_ratio_min_1":[[1,0]],
                  "conc_ratio_max_1":[[0,1]],
              }
    #with open(bc_filename,'rb') as infile:
    #    ceBulk = pck.load(infile)
    kwargs = {
      "crystalstructure":"fcc", "a":4.05, "size":[4,4,4], "basis_elements":[["Al","Mg"]],
      "conc_args":conc_args, "db_name":"dos-db.db",
    "max_cluster_size":4
    }
    ceBulk = BulkCrystal( **kwargs )
    print (ceBulk.basis_functions)

    eci_file = "data/ce_hydrostatic.json"
    with open( eci_file, 'r' ) as infile:
      ecis = json.load( infile )
    print (ecis)
    #calc = CE( ceBulk, ecis, size=(3,3,3) )
    calc = get_ce_calc( ceBulk, kwargs, ecis, size=[10,10,10] )
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )
    print ("Number of atoms: {}".format(len(ceBulk.atoms)) )

    T = 300
    dos = SGCCompositionFreeEnergy( nbins=10000, hist_limits=[("c1_0",0.45,1.0)])
    mc = sgc.SGCMonteCarlo( ceBulk.atoms, T, symbols=["Al","Mg"] )
    mc.chemical_potential = {"c1_0":-1.066}
    if ( run_array ):
        T = [150,200,250,300,350,375,390,400,410,420,430,440,450,460,470]
        mus = np.linspace(-1.064,-1.069,10)
        mu = [{"c1_0":value} for value in mus]
        array_runner = FreeEnergyMuTempArray( T, mu, fname="data/free_energy_al_rich.h5" )
        array_runner.run( mc, dos, min_num_steps=100000 )
    else:
        comp_free_eng = dos.find_dos(mc, min_num_steps=500000, max_rel_unc=0.01 )
        comp_free_eng.plot()
        plt.show()

if __name__ == "__main__":
    main()
