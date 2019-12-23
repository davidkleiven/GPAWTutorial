from ase.db import connect
from clease.mc_trajectory_extractor import MCTrajectoryExtractor
from clease.calculator import attach_calculator
from clease import settingFromJSON
import json
from matplotlib import pyplot as plt

db_name = 'almgsiX_dft2.db'
eci_file = "data/almgsiX_eci.json"

def find_swaps():
    all_atoms = []
    e_ref = []
    db = connect(db_name)
    for row in db.select(converged=True):
        all_atoms.append(row.toatoms())
        final_struct_id = row.final_struct_id
        final = db.get(id=final_struct_id)
        e_ref.append(final.energy)

    extractor = MCTrajectoryExtractor()
    swaps = extractor.find_swaps(all_atoms)

    with open(eci_file, 'r') as infile:
        eci = json.load(infile)

    e_pred = []
    setting = settingFromJSON("almgsixSettings.json")
    e_pred = {}
    for swap in swaps:
        print(swap)
        for s in swap:
            if s in e_pred.keys():
                continue
            atoms = attach_calculator(setting, atoms=all_atoms[s], eci=eci) 
            atoms.numbers = all_atoms[s].numbers
            e_pred[s] = atoms.get_potential_energy()

    for k, v in e_pred.items():
        print(k, v, e_ref[k])
    dev = extractor.swap_energy_deviations(swaps, e_pred, e_ref)
    extractor.plot_swap_deviation(dev)
    plt.show()

find_swaps()

    