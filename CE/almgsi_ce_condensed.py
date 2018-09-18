from ase.ce import BulkCrystal
from ase.ce import GenerateStructures
from ase.ce import Evaluate
import json

# File where ECIs are stored
eci_fname = "data/almgsi_fcc_eci_newconfig.json"

# Database used to have all the structures
db_name = "almgsi_newconfig.db"


def main():

    # Concentation arguments. NOTE: The way of specifying concentrations
    # will be changed in the future
    conc_args = {
        "conc_ratio_min_1": [[64, 0, 0]],
        "conc_ratio_max_1": [[24, 40, 0]],
        "conc_ratio_min_2": [[64, 0, 0]],
        "conc_ratio_max_2": [[22, 21, 21]]
    }

    ceBulk = BulkCrystal(
        crystalstructure="fcc", a=4.05, size=[4, 4, 4],
        basis_elements=[["Al", "Mg", "Si"]], conc_args=conc_args,
        db_name=db_name, max_cluster_size=4)

    # Create an instance of the structure generator
    struc_generator = GenerateStructures(ceBulk, struct_per_gen=10)

    # Generate new structures
    struc_generator.generate_probe_structure()

    # Evaluate and fit the ECIs
    # evaluate(ceBulk)


def evaluate(BC):
    # Set up an Evaluator with L1 regularization
    evaluator = Evaluate(BC, penalty="l1")

    # Try different penalization value to find the best
    best_alpha = evaluator.plot_CV(1E-5, 1E-3, num_alpha=16,
                                   logfile="almgsi_log.txt")

    # Find the ECIs using the best penalization value
    evaluator.plot_fit(best_alpha)
    print("Best penalization value: {}".format(best_alpha))
    eci_name = evaluator.get_cluster_name_eci(best_alpha, return_type="dict")

    with open(eci_fname, 'w') as outfile:
        json.dump(eci_name, outfile, indent=2, separators=(",", ":"))
    print("ECIs written to {}".format(eci_fname))


if __name__ == "__main__":
    main()
