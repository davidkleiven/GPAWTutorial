import sys
from clease import CEBulk, Concentration, NewStructures
from clease import Evaluate
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
import numpy as np


db_name = "almgsi_clease.db"


def main(argv):
    conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])
    kwargs = dict(crystalstructure="fcc", a=4.05, size=[4, 4, 4],
                  db_name=db_name, max_cluster_size=4, concentration=conc,
                  basis_function="sanchez", max_cluster_dia=[7.8, 5.0, 5.0])
    setting = CEBulk(**kwargs)
    # insert_from_db("almgsi.db", setting)
    evaluate(setting)


def insert_from_db(old_db_name, setting):
    db = connect(old_db_name)
    names = []
    for row in db.select(converged=1):
        names.append(row.name)

    ns = NewStructures(setting, struct_per_gen=10)
    for name in names:
        ignore = False
        print(name)
        init_struct = None
        final_struct = None
        energy = 0.0
        num_occ = sum(1 for _ in db.select(name=name))
        for row in db.select(name=name):
            if row["calculator"] == "unknown":
                energy = row.energy
                init_struct = row.toatoms()

                if num_occ != 2:
                    final_struct = row.toatoms()
            elif row["calculator"] == 'none':
                init_struct = row.toatoms()

                fid = row.get('final_struct_id', -1)
                if fid != -1:
                    final_struct = db.get(id=fid).toatoms()
                    energy = db.get(id=fid).get('energy', None)
                    if energy is None:
                        ignore = True
                    break
            else:
                final_struct = row.toatoms()

        calc = SinglePointCalculator(final_struct, energy=energy)
        final_struct.set_calculator(calc)
        try:
            ns.insert_structure(
                init_struct=init_struct,
                final_struct=final_struct,
                generate_template=True)
        except Exception as exc:
            print(str(exc))


def add_include_flag(db_name):
    db = connect(db_name)
    for row in db.select(converged=True):
        db.update(id=row.id, include=1)


def remove_high_si_content(db_name):
    db = connect(db_name)
    for row in db.select(include=1):
        atoms = row.toatoms()
        num_si = sum(1 for atom in atoms if atom.symbol == 'Si')

        if num_si/len(atoms) > 0.51:
            db.update(id=row.id, include=0)


def evaluate(setting):
    #add_include_flag("almgsi_clease.db")
    #remove_high_si_content("almgsi_clease.db")
    evaluator = Evaluate(setting, fitting_scheme='l1', parallel=False,
                         alpha=1E-6,
                         select_cond=[("converged", "=", True), ("include", "=", 1)],
                         scoring_scheme="k-fold")
    alpha = evaluator.plot_CV()
    evaluator.set_fitting_scheme(fitting_scheme="l1", alpha=alpha)
    evaluator.plot_fit(interactive=True)
    eci_name = evaluator.get_cluster_name_eci()

    e_dft = evaluator.e_dft
    e_pred = evaluator.cf_matrix.dot(evaluator.eci)
    concs = evaluator.concs
    al_conc = [x.get('Al', 0.0) for x in concs]
    mg_conc = [x.get('Mg', 0.0) for x in concs]
    si_conc = [x.get('Si', 0.0) for x in concs]
    data = np.vstack((al_conc, mg_conc, si_conc, e_pred, e_dft)).T
    #np.savetxt("data/almgsi_ce.csv", data, delimiter=',', header='Al, Mg, Si, E_ce, E_dft')

if __name__ == '__main__':
    main(sys.argv[1:])