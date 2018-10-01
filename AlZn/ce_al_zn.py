from ase.ce import BulkCrystal, Evaluate, CorrFunction, GenerateStructures
from atomtools.ce import ECIPlotter
from matplotlib import pyplot as plt
import json


def main():
    alat = 3.9 # Something in between Cu and Au
    kwargs_fcc = {
        "crystalstructure": "fcc",
        "a": alat,
        "size": [2, 2, 2],
        "basis_elements": [["Au", "Cu"]],
        "conc_args": {"conc_ratio_min_1": [[1, 0]],
                      "conc_ratio_max_1": [[0, 1]]},
        "max_cluster_size": 4,
        "db_name": "data/au_cu.db"
    }
    fcc = BulkCrystal(**kwargs_fcc)
    gen_struct(fcc)
    
def gen_struct(bc):
    struct_gen = GenerateStructures(bc)
    struct_gen.generate_initial_pool()


def evaluate(bc, lattice):
    # reconfigure(bc)

    eval_fcc = Evaluate(bc, penalty="l1")
    min_alpha = eval_fcc.plot_CV(1E-10, 1E-2, num_alpha=80)
    eci_dict = eval_fcc.get_cluster_name_eci(min_alpha, return_type='dict')
    eci_plotter = ECIPlotter(eci_dict)
    eci_plotter.plot()
    fname = "data/eci_alzn_{}.json".format(lattice)
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
