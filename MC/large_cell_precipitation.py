import sys
from cemc.mcmc import Montecarlo
from ase.ce import BulkCrystal
import json
from cemc.wanglandau.ce_calculator import get_ce_calc
from cemc.mcmc import Snapshot, PairCorrelationObserver, NetworkObserver
import json
from matplotlib import pyplot as plt
from ase.visualize import view
from ase.io.trajectory import Trajectory
import numpy as np
from ase.io import write
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

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
    calc = get_ce_calc( ceBulk, kwargs, ecis, size=[15,15,15], free_unused_arrays_BC=True )
    ceBulk = calc.BC
    ceBulk.atoms.set_calculator( calc )
    comp = {
        "Al":1.0-mg_conc,
        "Mg":mg_conc
    }
    calc.set_composition(comp)
    mc_obj = Montecarlo( ceBulk.atoms, T, mpicomm=comm )
    pairs = PairCorrelationObserver( calc )
    network = NetworkObserver( calc=calc, cluster_name="c2_1414_1", element="Mg", nbins=100 )
    mode = "fixed"
    camera = Snapshot( "data/precipitation.traj", atoms=ceBulk.atoms )
    mc_obj.attach( camera, interval=10000 )
    mc_obj.attach( pairs, interval=1 )
    mc_obj.attach( network, interval=500 )
    mc_obj.runMC( mode=mode, steps=40000000, equil=True )

    pair_mean = pairs.get_average()
    pair_std = pairs.get_std()

    if ( rank == 0 ):
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
        atoms = network.get_atoms_with_largest_cluster()

    data = network.get_statistics() # This collects the histogram data from all processors
    size,occurence = network.get_size_histogram()
    data["histogram"] = {}
    data["histogram"]["size"] = size.tolist()
    data["histogram"]["occurence"] = occurence.tolist()
    data["temperature"] = T
    data["mg_conc"] = mg_conc

    cluster_fname = "data/cluster_statistics_{}_{}K.json".format(int(1000*mg_conc),int(T))
    atoms_fname = "data/largest_cluster_{}_{}K.cif".format( int(1000*mg_conc), int(T) )

    if ( rank == 0 ):
        print (occurence)

    if ( rank == 0 ):
        try:
            with open( cluster_fname,'r') as infile:
                data = json.load(infile)
            old_size = np.array(data["histogram"]["size"])
            old_hist = np.array(data["histogram"]["occurence"])
            if ( np.allclose(old_size,size) ):
                occurence += old_hist
                data["histogram"]["occurence"] = occurence.tolist()
        except Exception as exc:
            print (str(exc))

        with open( cluster_fname, 'w' ) as outfile:
            json.dump( data, outfile, indent=2, separators=(",",":") )
        print ("Cluster statistics written to {}".format(cluster_fname) )
        write( atoms_fname, atoms )
        print ("Atoms with largest cluster written to {}".format(atoms_fname))
    #view(atoms)
    #plt.plot( network.size_histogram, ls="steps")
    #plt.show()
    print ("Proc: {} reached final barrier".format(rank))
    comm.barrier()

def highlight_snn_mg(atoms):
    atoms_in_cluster = []
    indx = range(len(atoms))
    for i in range(len(atoms)):
        if ( atoms[i].symbol != "Mg" ):
            continue
        dists = atoms.get_distances(i,indx,mic=True)
        for j,d in enumerate(dists):
            if ( np.abs(d-4.96) < 0.001 and atoms[j].symbol == "Mg"):
                if ( i not in atoms_in_cluster ):
                    atoms_in_cluster.append(i)
                if ( j not in atoms_in_cluster ):
                    atoms_in_cluster.append(j)

    for indx in atoms_in_cluster:
        atoms[indx].symbol = "Na"
    view(atoms)

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

def highlight_pairs():
    traj = Trajectory("data/precipitation.traj")
    atoms = traj[-1]
    highlight_snn_mg(atoms)

def main( argv ):
    T = int(argv[0])
    run(T,0.05)
    #plot_pairs()
    #highlight_pairs()

if __name__ == "__main__":
    main( sys.argv[1:] )
