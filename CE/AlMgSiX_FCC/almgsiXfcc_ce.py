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
    N = 4
    conc = Concentration(basis_elements=[["Al", "Mg", "Si", "X"]])
    kwargs = dict(crystalstructure="fcc", a=4.05, size=[N,N,N],
        db_name=db_name, max_cluster_size=4, max_cluster_dia=[0.0, 0.0, 5.0, 4.1, 4.1],
        concentration=conc)
    ceBulk = BulkCrystal(**kwargs)

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
    elif option == "newdb":
        new_db_name = argv[1]
        new_db(kwargs, new_db_name)
    elif option == "reconfig_db":
        corr_func = CorrFunction(ceBulk, parallel=True)
        scond = [("calculator", "!=", "gpaw"),
                 ("name", "!=", "information"),
                 ("name", "!=", "template"),
                 ("converged", "=", 1)]
        corr_func.reconfigure_db_entries(select_cond=scond)
    elif option == "gs":
        get_gs_allgs(ceBulk, struc_generator)

def new_db(kwargs, new_db_name):
    from ase.db import connect
    from ase.calculators.singlepoint import SinglePointCalculator
    old_db_name = kwargs["db_name"]
    kwargs["db_name"] = new_db_name
    ceBulk = BulkCrystal(**kwargs)
    struc_generator = GenerateStructures( ceBulk, struct_per_gen=10 )
    names = []
    db = connect(old_db_name)
    for row in db.select(converged=1):
        names.append(row.name)
    
    for name in names:
        row = db.get(name=name, struct_type='initial')
        calc = row.get("calculator", "")
        energy = None
        init_struct = row.toatoms()
        if calc == "unknown":
            energy = row.energy

        if calc == "":
            final_struct = db.get(id=row.final_struct_id).toatoms()
        else:
            try:
                final_struct = db.get(name=name, struct_type='final').toatoms()
            except KeyError:
                final_struct = db.get(name=name, calculator='gpaw').toatoms()
        
        if energy is not None:
            sp = SinglePointCalculator(final_struct, energy=energy)
            final_struct.set_calculator(sp)
        
        assert final_struct.get_calculator() is not None
        try:
            struc_generator.insert_structure(init_struct=init_struct, final_struct=final_struct)
        except Exception as exc:
            print(str(exc))


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
    atoms = ceBulk.atoms.copy()
    calc = CE(atoms, ceBulk, eci=eci)
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
        symbs[2] = "X"
        symbs[3] = "X"
        calc.set_symbols(symbs)

        temps = np.linspace(1.0, 2000.0, 30)[::-1]
        nsteps = 10*len(atoms)
        gs = GSFinder()
        res = gs.get_gs(ceBulk, temps=temps, n_steps_per_temp=nsteps, atoms=atoms)
        try:
            struct_gen.insert_structure(init_struct=res["atoms"])
        except Exception as exc:
            print(str(exc))


def evaluate(bc):
    from ase.db import connect
    #scond = [("converged", "=", True)]
    # db = connect(bc.db_name)
    # for row in db.select(scond):
    #     final = row.get("final_struct_id", -1)
    #     energy = row.get("energy", 0)
    #     if final == -1 and energy == 0:
    #         print(row.id, row.name)
    #         db.update(row.id, converged=0)
    # exit()
    scond = [("calculator", "!=", "gpaw"),
             ("name", "!=", "information"),
             ("name", "!=", "template"),
             ("converged", "=", 1)]
    evaluator = Evaluate(bc, select_cond=scond, scoring_scheme="loocv_fast")
    ga_fit = GAFit(setting=bc, alpha=1E-8, change_prob=0.2, select_cond=scond, fname="data/ga_almgsiX.csv")
    ga_fit.run(min_change=0.001)
    eci = evaluator.get_cluster_name_eci()
    #evaluator.plot_fit()

    with open(ECI_FILE, 'w') as outfile:
        json.dump(eci, outfile, indent=2, separators=(",", ": "))
    print("ECI written to {}".format(ECI_FILE))

if __name__ == "__main__":
    main(sys.argv[1:])
