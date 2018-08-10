from ase.ce import BulkCrystal, Evaluate, CorrFunction, GenerateStructures
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from atomtools.ce import ECIPlotter
from matplotlib import pyplot as plt
import json
import numpy as np
from ase.io import write
import time

def main():
    alat = 3.8
    conc_args = {}
    conc_args['conc_ratio_min_1'] = [[1, 0]]
    conc_args['conc_ratio_max_1'] = [[0, 1]]
    kwargs_fcc = {
        "crystalstructure": 'fcc',
        "a": 3.8,
        "size": [4, 4, 2],
        # "size": [2, 2, 2],
        "basis_elements": [['Cu', 'Au']],
        "conc_args": conc_args,
        "db_name": 'cu-au_fcc_final.db',
        "max_cluster_size": 3,
        "max_cluster_dist": 1.5*alat
        }

    # alat = 3.9 # Something in between Cu and Au
    # kwargs_fcc = {
    #     "crystalstructure": "fcc",
    #     "a": alat,
    #     "size": [2, 2, 4],
    #     "basis_elements": [["Au", "Cu"]],
    #     "conc_args": {"conc_ratio_min_1": [[1, 0]],
    #                   "conc_ratio_max_1": [[0, 1]]},
    #     "max_cluster_size": 4,
    #     "max_cluster_dist": 1.05*alat,
    #     "db_name": "data/au-cu_fcc.db"
    # }
    fcc = BulkCrystal(**kwargs_fcc)
    # fcc.reconfigure_settings()
    # gen_struct(fcc)
    # gs_struct(fcc)
    evaluate(fcc)

def gen_struct(bc):
    struct_gen = GenerateStructures(bc)
    # struct_gen.generate_initial_pool()
    # struct_gen.generate_probe_structure(num_steps=100, num_samples_var=10)

def gs_struct(bc):
    from cemc.tools import GSFinder
    from ase.visualize import view
    with open("data/eci_aucu.json", 'r') as infile:
        eci = json.load(infile)
    struct_gen = GenerateStructures(bc, struct_per_gen=20)
    conc = np.arange(0.25, 0.75)
    conc = [0.25, 0.75]
    for au_conc in conc:
        comp = {"Au": au_conc, "Cu": 1.0 - au_conc}
        T = [800, 700, 600, 500, 400, 300, 200, 100, 50, 20, 10]
        nsteps_per_temp = 100 * len(bc.atoms)
        finder = GSFinder()
        gs = finder.get_gs(bc, eci, composition=comp, n_steps_per_temp=nsteps_per_temp, temps=T)
        try:
            # view(gs["atoms"])
            # struct_gen.insert_structure(init_struct=gs["atoms"])
            fname = "Emin_structure_{}_{}.xyz".format(int(au_conc*100), time.time())
            write(fname, gs["atoms"])
        except Exception as exc:
            print(exc)


def evaluate(bc):
    # reconfigure(bc)

    eval_fcc = Evaluate(bc, penalty="l1")
    min_alpha = eval_fcc.plot_CV(1E-12, 1E-3, num_alpha=200)
    # min_alpha = 1E-6
    eval_fcc.plot_fit(min_alpha)
    eci_dict = eval_fcc.get_cluster_name_eci(min_alpha, return_type='dict')
    eci_plotter = ECIPlotter(eci_dict, naming="normalized")
    eci_plotter.plot()
    fname = "data/eci_aucu.json"
    with open(fname, 'w') as outfile:
        json.dump(eci_dict, outfile, indent=2, separators=(",", ":"))
    print("ECI written to {}".format(fname))
    plt.show()


def reconfigure(bc):
    bc.reconfigure_settings()
    corr_func = CorrFunction(bc)
    corr_func.reconfig_db_entries(reset=True)

if __name__ == "__main__":
    main()
