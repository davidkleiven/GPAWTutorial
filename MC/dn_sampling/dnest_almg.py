import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
from cemc.dns import DNSampler
from cemc.mcmc import SGCMonteCarlo
from ase.ce import BulkCrystal
import json
from cemc.wanglandau.ce_calculator import get_ce_calc
from matplotlib import pyplot as plt

def main():
    chem_pots = {
        "c1_0":-1.067
    }

    conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
    #with open(bc_filename,'rb') as infile:
    #    ceBulk = pck.load(infile)
    kwargs = {
        "crystalstructure":"fcc", "a":4.05, "size":[4,4,4], "basis_elements":[["Al","Mg"]],
        "conc_args":conc_args, "db_name":"random_test.db",
     "max_cluster_size":4
    }
    ceBulk = BulkCrystal( **kwargs )
    print (ceBulk.basis_functions)

    eci_file = "../data/ce_hydrostatic.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    print (ecis)
    #calc = CE( ceBulk, ecis, size=(3,3,3) )
    calc = get_ce_calc( ceBulk, kwargs, ecis, size=[10,10,10] )
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )
    print ("Number of atoms: {}".format(len(ceBulk.atoms)) )
    mc = SGCMonteCarlo( ceBulk.atoms, 100000, symbols=["Al","Mg"] )
    mc.chemical_potential = chem_pots
    dn_sampler = DNSampler(mc_sampler=mc)
    dnest4_args = {"max_num_levels":50,"lam":10}
    dn_sampler.run(dnest4_args=dnest4_args)
    dn_sampler.make_histograms(bins=100)
    plt.show()

if __name__ == "__main__":
    main()
