import sys
from ase.clease import CEBulk as BulkCrystal
from ase.clease import Concentration 
from ase.clease import CorrFunction
from ase.clease import NewStructures as GenerateStructures
from ase.clease import GAFit, Evaluate
from cemc.tools import GSFinder
from cemc import CE
import json


db_name = "almgsiX_fcc.db"
ECI_FILE = "data/eci_almgsix.json"

def main(argv):
    option = argv[0]
    conc_args = {
        "conc_ratio_min_1":[[64,0,0]],
        "conc_ratio_max_1":[[24,40,0]],
        "conc_ratio_min_2":[[64,0,0]],
        "conc_ratio_max_2":[[22,21,21]]
    }
    N = 4
    conc = Concentration(basis_elements=[["Al", "Mg", "Si", "X"]])
    ceBulk = BulkCrystal(crystalstructure="fcc", a=4.05, size=[N,N,N],
        db_name=db_name, max_cluster_size=4, max_cluster_dia=[0.0, 0.0, 5.0, 4.1, 4.1],
        concentration=conc)

    struc_generator = GenerateStructures( ceBulk, struct_per_gen=10 )
    if option == "reconfig_settings":
        ceBulk.reconfigure_settings()
    elif option == "insert":
        fname = argv[1]
        struc_generator.insert_structure(init_struct=fname)
    elif option == "random":
        atoms = get_random_structure()
        struc_generator.insert_structure(init_struct=atoms)
    elif option == "evaluate":
        evaluate(ceBulk)
    elif option == "reconfig_db":
        corr_func = CorrFunction(ceBulk, parallel=True)
        scond = [("calculator", "!=", "gpaw"),
                 ("name", "!=", "information"),
                 ("name", "!=", "template")]
        corr_func.reconfig_db_entries(select_cond=scond)
    elif option == "gs":
        get_gs_allgs(ceBulk, struc_generator)


def get_random_structure():
    from ase.build import bulk
    from random import choice
    atoms = bulk("Al")*(4, 4, 4)
    symbs = ["Al", "Mg", "Si"]

    for atom in atoms:
        atom.symbol = choice(symbs)

    indx1 = choice(range(len(atoms)))
    indx2 = choice(range(len(atoms)))
    atoms[indx1].symbol = "X"
    atoms[indx2].symbol = "X"
    return atoms

def gs_compositions():
    from itertools import product
    al_conc = [0.0, 0.125, 0.375, 0.5, 0.625, 0.75, 0.875]
    concs = []
    for mgal_conc in product(al_conc, repeat=2):
        new_conc = [mgal_conc[0], mgal_conc[1], 1.0 - mgal_conc[0] - mgal_conc[1]]
        concs.append(new_conc)
    return concs
    

def get_gs_allgs(ceBulk, struct_gen):
    import numpy as np
    from random import shuffle
    with open(ECI_FILE, 'r') as infile:
        eci = json.load(infile)
    calc = CE(ceBulk, eci=eci)
    atoms = ceBulk.atoms
    atoms.set_calculator(calc)
    symbs = ["Al"]*len(atoms)
    for conc in gs_compositions():
        num_al = conc[0]*len(atoms)
        num_mg = conc[1]*len(atoms)
        for i in range(len(symbs)):
            if i < num_al:
                symbs[i] = "Al"
            elif i < num_al+num_mg:
                symbs[i] = "Mg"
            else:
                symbs[i] = "Si"
        shuffle(symbs)
        symbs[0] = "X"
        symbs[1] = "X"
        calc.set_symbols(symbs)

        temps = np.linspace(1.0, 2000.0, 30)[::-1]
        nsteps = 10*len(atoms)
        gs = GSFinder()
        res = gs.get_gs(ceBulk, temps=temps, n_steps_per_temp=nsteps)
        try:
            struct_gen.insert_structure(init_struct=res["atoms"])
        except Exception as exc:
            print(str(exc))


def evaluate(bc):
    from ase.db import connect
    scond = [("converged", "=", True)]
    # db = connect(bc.db_name)
    # for row in db.select(scond):
    #     final = row.get("final_struct_id", -1)
    #     energy = row.get("energy", 0)
    #     if final == -1 and energy == 0:
    #         print(row.id, row.name)
    #         db.update(row.id, converged=0)
    # exit()
    evaluator = Evaluate(bc, select_cond=scond, scoring_scheme="loocv_fast")
    ga_fit = GAFit(evaluator=evaluator, alpha=1E-8, change_prob=0.2)
    ga_fit.run(min_change=0.001)
    eci = evaluator.get_cluster_name_eci()
    evaluator.plot_fit()

    with open(ECI_FILE, 'w') as outfile:
        json.dump(eci, outfile, indent=2, separators=(",", ": "))
    print("ECI written to {}".format(ECI_FILE))

if __name__ == "__main__":
    main(sys.argv[1:])
