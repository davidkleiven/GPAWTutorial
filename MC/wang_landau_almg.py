import sys
from cemc.wanglandau import WangLandauInit, WangLandau, WangLandauDBManager
import json
import numpy as np

wl_db = "data/wang_landau_almg.db"
eci_fname = "data/ce_hydrostatic.json"

bc_kwargs = {
    "crystalstructure":"fcc",
    "size":[4,4,4],
    "basis_elements":[["Al","Mg"]],
    "db_name":"temporary_bcdb.db",
    "conc_args":{"conc_ratio_min_1":[[1,0]],"conc_ratio_max_1":[[0,1]]},
    "max_cluster_dia":4,
    "a":4.05
}

with open(eci_fname,'r') as infile:
    eci = json.load(infile)


def insert_new_atoms( comp ):
    initializer = WangLandauInit( wl_db )
    T = np.logspace(0,3,50)[::-1]
    try:
        initializer.insert_atoms( bc_kwargs, size=[10,10,10], T=T, n_steps_per_temp=10000, eci=eci, composition=comp )
    except Exception as exc:
        print (str(exc))
        pass
    initializer.prepare_wang_landau_run([("id","=","1")])

def run( atomID ):
    initializer = WangLandauInit( wl_db )
    atoms = initializer.get_atoms( atomID, eci )
    db_manager = WangLandauDBManager( wl_db )
    runID = db_manager.get_next_non_converged_uid(atomID)
    if ( runID == -1 ):
        print ("RunID=-1")
        return
    simulator = WangLandau( atoms, wl_db, runID, fmin=1E-4, Nbins=500, scheme="inverse_time" )
    simulator.run_fast_sampler( mode="adaptive_windows", maxsteps=int(1E9) )
    simulator.save_db()

def main( argv ):
    option = argv[0]
    if ( option == "insert" ):
        mg_conc = float(argv[1])
        comp = {
            "Al":1-mg_conc,
            "Mg":mg_conc
        }
        insert_new_atoms( comp )
    elif ( option == "run" ):
        atomID = int(argv[1])
        run(atomID)

if __name__ == "__main__":
    main( sys.argv[1:] )
