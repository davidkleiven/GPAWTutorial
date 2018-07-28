import sys
sys.path.insert(1,"/home/davidkl/Documents/aseJin")

from cemc import get_ce_calc
from cemc.mcmc import Snapshot, Montecarlo
from ase.ce import BulkCrystal
import json

eci_file = "data/almgsi_fcc_eci_newconfig.json"
traj_file = "data/huge_cell_50x50x50.traj"
def run_huge():
    conc_args = {
        "conc_ratio_min_1":[[64,0,0]],
        "conc_ratio_max_1":[[24,40,0]],
        "conc_ratio_min_2":[[64,0,0]],
        "conc_ratio_max_2":[[22,21,21]]
    }

    kwargs = {
        "crystalstructure":"fcc",
        "size":[4,4,4],
        "basis_elements":[["Al","Mg","Si"]],
        "conc_args":conc_args,
        "db_name":"some_db.db",
        "max_cluster_size":4,
        "a":4.05,
        "ce_init_alg":"fast"
    }
    ceBulk = BulkCrystal( **kwargs )

    with open(eci_file, 'r') as infile:
        eci = json.load(infile)
    calc = get_ce_calc(ceBulk, kwargs, eci=eci, size=[50,50,50])
    calc.BC.atoms.set_calculator(calc)
    print(calc.BC.basis_functions)

    comp = {
        "Al":0.9,
        "Mg":0.05,
        "Si":0.05
    }
    calc.set_composition(comp)

    T = 150
    T = [2000, 1500, 1200, 1000, 800, 600, 400, 293]
    camera = Snapshot(trajfile=traj_file, atoms=calc.BC.atoms)
    for temp in T:
        print("Current temperature {}K".format(temp))
        mc = Montecarlo(calc.BC.atoms, temp)
        mc.attach(camera, interval=125000*20)
        mc.runMC(mode="fixed", steps=125000*100, equil=False)

def main():
    run_huge()
if __name__ == "__main__":
    main()
