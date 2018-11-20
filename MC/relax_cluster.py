import sys
sys.path.insert(1,"/home/davidkl/Documents/ase-ce0.1")
sys.path.insert(2,"/home/dkleiven/Documents/aseJin")
from ase.clease import CEBulk as BulkCrystal
from ase.clease import Concentration
from cemc import get_ce_calc
from cemc.mcmc import FixedNucleusMC
from cemc.mcmc import LowestEnergyStructure
import json
from ase.visualize import view
import numpy as np
from ase.io import write

# Define some global parameters
conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
conc = Concentration(basis_elements=[["Al", "Mg"]])
kwargs = {
    "crystalstructure":"fcc", "a":4.05, "size":[4,4,4], "concentration":conc,
    "db_name":"data/temporary_bcnucleationdb.db",
    "max_cluster_size":4
}

eci_file = "data/ce_hydrostatic_only_relaxed.json"
with open( eci_file, 'r' ) as infile:
    ecis = json.load( infile )

def get_pure_energies(eci):
    al = 0.0
    mg = 0.0
    for key,value in eci.items():
        al += value
        if ( int(key[1])%2 == 0 ):
            mg += value
        else:
            mg -= value
    return al,mg

folder = "data/cluster_struct_new_trialmove/"
folder = "/work/sophus/cluster_size_free_energy"

def cluster_entropy(mc, size=8):
    from ase.io import read
    from cemc.mcmc import EnergyEvolution
    import dataset
    from cemc.mcmc import Snapshot
    T = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 150, 200, 220, 240, 260, 280, 300,
         320, 340, 360, 380, 400, 420, 440, 460, 480, 500]
    
    atoms = read("data/cluster_struct_new_trialmove/cluster{}_all.cif".format(size))
    symbs = [atom.symbol for atom in atoms]
    mc.set_symbols(symbs)
    energy_evol = EnergyEvolution(mc)
    mc.attach(energy_evol, interval=10000)
    db = dataset.connect("sqlite:///{}/heat_almg_cluster_size.db".format(folder, size))
    syst = db["thermodynamic"]
    energy_evol_tab = db["energy_evolution"]
    camera = Snapshot(trajfile=folder+"/cluster_entropy_size{}.traj".format(size), atoms=mc.atoms)
    mc.attach(camera, interval=50000)
    for temp in T:
        print("Temperature: {}".format(temp))
        mc.reset()
        energy_evol.reset()
        mc.T = temp
        mc.runMC(steps=500000)
        thermo = mc.get_thermodynamic()
        thermo["size"] = size
        uid = syst.insert(thermo)
        rows = []
        for E in energy_evol.energies:
            rows.append({"uid": uid, "energy": E})
        energy_evol_tab.insert_many(rows)

def pure_phase_entropy(small_bc):
    from cemc.mcmc import Montecarlo
    from ase.clease.tools import wrap_and_sort_by_position
    from ase.io import read
    from cemc.mcmc import SiteOrderParameter, Snapshot
    import dataset
    T = [1, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300, 310, 
         320, 330, 340, 350, 360, 370, 380, 390, 400, 420, 440, 460, 480, 500]
    calc = get_ce_calc(small_bc, kwargs, ecis, size=[10, 10, 10], db_name="data/db10x10x10Al3Mg.db")
    bc = calc.BC
    bc.atoms.set_calculator(calc)
    atoms = read("data/al3mg_template.xyz")
    atoms = wrap_and_sort_by_position(atoms)
    symbs = [atom.symbol for atom in atoms]
    calc.set_symbols(symbs)
    
    site_order = SiteOrderParameter(bc.atoms)
    db = dataset.connect("sqlite:///data/pure_al3mg3.db")
    syst = db["thermodynamic"]
    camera = Snapshot(trajfile="data/pure_phase_entropy.traj", atoms=bc.atoms)
    for temp in T:
        print("Current temperature: {}K".format(temp))
        site_order.reset()
        mc = Montecarlo(bc.atoms, temp)
        mc.attach(site_order)
        mc.attach(camera, interval=50000)
        equil_param = {"window_length": 100000, "mode": "fixed"}
        mc.runMC(mode="fixed", steps=500000, equil=True, equil_params=equil_param)
        mean, stddev = site_order.get_average()
        thermo = mc.get_thermodynamic()
        thermo["site_order"] = mean
        thermo["site_order_std"] = stddev
        syst.insert(thermo)

def main(option="relax", size=8):
    from copy import deepcopy
    ceBulk = BulkCrystal( **kwargs )
    bc_copy = deepcopy(ceBulk)

    print (ecis)
    al,mg = get_pure_energies(ecis)
    print ("Al energy: {} eV/atom".format(al))
    print ("Mg energy: {} eV/atom".format(mg))
    #exit()
    #calc = CE( ceBulk, ecis, size=(3,3,3) )
    calc = get_ce_calc(ceBulk, kwargs, ecis, size=[15,15,15])
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )
    #pure_energy = calc.get_energy()
    #print (pure_energy)
    #energy = calc.calculate( ceBulk.atoms, ["energy"],[(0,"Al","Mg")])
    #print (energy)
    #energy = calc.calculate( ceBulk.atoms, ["energy"], [()])
    #exit()

    if option == "heat":
        print("Running with cluser size {}".format(size))
        mc = FixedNucleusMC( ceBulk.atoms, 293, network_name=["c2_4p050_3"], network_element=["Mg"] )
        cluster_entropy(mc, size=size)
        return
    elif option == "pure_phase":
        pure_phase_entropy(bc_copy)
        return
    else:
        sizes = range(3,51)
        #sizes = np.arange(3, 51)
        energies = []
        cell = ceBulk.atoms.get_cell()
        diag = 0.5*(cell[0, :] + cell[1, :] + cell[2, :])
        pos = ceBulk.atoms.get_positions()
        pos -= diag
        lengths = np.sum(pos**2, axis=1)
        indx = np.argmin(lengths)
        symbs = [atom.symbol for atom in ceBulk.atoms]
        symbs[indx] = "Mg"
        print("Orig energy: {}".format(calc.get_energy()))
        calc.set_symbols(symbs)
        print("One atom: {}".format(calc.get_energy()))
        exit()

        for size in sizes:
            elements = {"Mg": size}
            T = np.linspace(50,1000,40)[::-1]

            # Reset the symbols to only Mg
            calc.set_symbols(symbs)
            mc = FixedNucleusMC( ceBulk.atoms, T, network_name=["c2_4p050_3"], network_element=["Mg"] )
            mc.grow_cluster(elements)
            mc.current_energy = calc.get_energy()
            low_en = LowestEnergyStructure( calc, mc, verbose=True )
            mc.attach( low_en )
            for temp in T:
                print ("Temperature {}K".format(temp))
                mc.T = temp
                mc.runMC(steps=10000, init_cluster=False)
                mc.reset()
                mc.is_first = False
            # atoms,clust = mc.get_atoms( atoms=low_en.atoms )
            write( "{}cluster{}_all.cif".format(folder,size), low_en.atoms )
            # write( "{}cluster{}_cluster.cif".format(folder,size), clust )
            energies.append(low_en.lowest_energy)
            print (sizes,energies)
        data = np.vstack((sizes,energies)).T
        np.savetxt( "{}energies_run2.txt".format(folder), data, delimiter=",")

if __name__ == "__main__":
    import sys
    size = 8
    for arg in sys.argv:
        if "--size=" in arg:
            size = int(arg.split("--size=")[1])
    main(option="heat", size=size)
