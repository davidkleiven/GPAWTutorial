import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
sys.path.insert(2,"/home/dkleiven/Documents/aseJin")
from ase.ce import BulkCrystal
from cemc.wanglandau.ce_calculator import get_ce_calc
from cemc.mcmc import FixedNucleusMC
from cemc.mcmc import LowestEnergyStructure
import json
from ase.visualize import view
import numpy as np
from ase.io import write

def get_pure_energies(eci):
    al = 0.0
    mg = 0.0
    for key,value in eci.iteritems():
        al += value
        if ( int(key[1])%2 == 0 ):
            mg += value
        else:
            mg -= value
    return al,mg

folder = "data/cluster_struct_longrun/"
def main():
    conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
    kwargs = {
        "crystalstructure":"fcc", "a":4.05, "size":[4,4,4], "basis_elements":[["Al","Mg"]],
        "conc_args":conc_args, "db_name":"data/temporary_bcnucleationdb.db",
        "max_cluster_size":4
    }
    ceBulk = BulkCrystal( **kwargs )
    print (ceBulk.basis_functions)

    eci_file = "data/ce_hydrostatic.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    print (ecis)
    al,mg = get_pure_energies(ecis)
    print ("Al energy: {} eV/atom".format(al))
    print ("Mg energy: {} eV/atom".format(mg))
    #exit()
    #calc = CE( ceBulk, ecis, size=(3,3,3) )
    calc = get_ce_calc( ceBulk, kwargs, ecis, size=[15,15,15], free_unused_arrays_BC=False)
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )
    #pure_energy = calc.get_energy()
    #print (pure_energy)
    #energy = calc.calculate( ceBulk.atoms, ["energy"],[(0,"Al","Mg")])
    #print (energy)
    #energy = calc.calculate( ceBulk.atoms, ["energy"], [()])
    #exit()

    sizes = range(3,51)
    energies = []
    for size in sizes:
        T = np.logspace(3,-1,100)
        mc = FixedNucleusMC( ceBulk.atoms, T, size=size, network_name="c2_1414_1", network_element="Mg" )
        low_en = LowestEnergyStructure( calc, mc )
        mc.attach( low_en )
        init_cluster = True
        for temp in T:
            print ("Temperature {}K".format(temp))
            mc.T = temp
            mc.run( nsteps=100000, init_cluster=init_cluster )
            init_cluster = False
            mc.reset()
        atoms,clust = mc.get_atoms( atoms=low_en.atoms )
        write( "{}cluster{}_all.cif".format(folder,size), atoms )
        write( "{}cluster{}_cluster.cif".format(folder,size), clust )
        energies.append(low_en.lowest_energy)
        print (sizes,energies)
    data = np.vstack((sizes,energies)).T
    np.savetxt( "{}energies_run2.txt".format(folder), data, delimiter=",")

if __name__ == "__main__":
    main()
