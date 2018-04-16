from cemc.mcmc import Montecarlo
from ase.ce import BulkCrystal
import json
from cemc.wanglandau.ce_calculator import get_ce_calc
from cemc.mcmc import Snapshot, PairCorrelationObserver
import json
from matplotlib import pyplot as plt

def run(T,mg_conc):
    conc_args = {
                "conc_ratio_min_1":[[1,0]],
                "conc_ratio_max_1":[[0,1]],
            }
    kwargs = {
        "crystalstructure":"fcc", "a":4.05, "size":[4,4,4], "basis_elements":[["Al","Mg"]],
        "conc_args":conc_args, "db_name":"data/temporary_bcdb.db",
        "max_cluster_size":4
    }
    ceBulk = BulkCrystal( **kwargs )
    print (ceBulk.basis_functions)

    eci_file = "data/ce_hydrostatic.json"
    with open( eci_file, 'r' ) as infile:
        ecis = json.load( infile )
    print (ecis)
    #calc = CE( ceBulk, ecis, size=(3,3,3) )
    calc = get_ce_calc( ceBulk, kwargs, ecis, size=[15,15,15] )
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )
    comp = {
        "Al":1.0-mg_conc,
        "Mg":mg_conc
    }
    calc.set_composition(comp)
    mc_obj = Montecarlo( ceBulk.atoms, T )
    pairs = PairCorrelationObserver( calc )
    mode = "fixed"
    camera = Snapshot( "data/precipitation.traj", atoms=ceBulk.atoms )
    mc_obj.attach( camera, interval=10000 )
    mc_obj.attach( pairs, interval=1 )
    mc_obj.runMC( mode=mode, steps=500000, equil=True )

    pair_mean = pairs.get_average()
    pair_std = pairs.get_std()

    data = {
        "pairs":pair_mean,
        "pairs_std":pair_std,
        "mg_conc":mg_conc,
        "temperature":T
    }
    pairfname = "data/precipitation_pairs/precipitation_pairs_{}_{}K.json".format(int(1000*mg_conc),int(T))
    with open(pairfname,'w') as outfile:
        json.dump(data,outfile,indent=2, separators=(",",":"))
    print ( "Thermal averaged pair correlation functions written to {}".format(pairfname) )

def plot_pairs():
    pairs350 = [50,100,150,200,250]
    cname = "c2_1414_1_00"
    concs350 = []
    mean350 = []
    std350 = []
    for ending in pairs350:
        fname = "data/precipitation_pairs/precipitation_pairs_{}_350K.json".format(ending)
        with open( fname, 'r' ) as infile:
            data = json.load(infile)
        mean = data["pairs"][cname]
        std = data["pairs_std"][cname]
        mg_conc = data["mg_conc"]
        concs350.append( mg_conc )
        x = (1.0-2.0*mg_conc)/2.0
        #mean /= x**2
        #std /= x**2
        mean350.append( mean )
        std350.append( std )

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.errorbar( concs350, mean350, std350 )
    ax.set_xlabel( "Mg concentration" )
    ax.set_ylabel( "\$\phi_\mathrm{snn}\$" )
    plt.show()

def main():
    #run(350,0.05)
    plot_pairs()

if __name__ == "__main__":
    main()
