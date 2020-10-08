from clease.settings import CEBulk
from clease.tools import reconfigure
from clease.settings import Concentration
from clease import NewStructures
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
db_name = "almgsi_new.db"
db_name_old = "almgsi.db"
def reconfigure_almgsi(): 
    conc = Concentration(basis_elements=[["Al", "Mg", "Si"]])
    kwargs = dict(crystalstructure="fcc", a=4.05, size=[1, 1, 1], \
        db_name=db_name, max_cluster_size=4, concentration=conc, max_cluster_dia=[7.8, 5.0, 5.0])
    ceBulk = CEBulk(**kwargs)

    db = connect(db_name_old)
    counter = 0
    new_struct = NewStructures(ceBulk)
    for row in db.select([('converged', '=', 1)]):
        print("ROW: {}".format(counter))
        counter += 1
        init = row.toatoms()
        final = init.copy()
        energy = row.get('energy', None)
        if energy is None:
            final_struct_id = row.final_struct_id
            energy = db.get(id=final_struct_id).energy
        calc = SinglePointCalculator(final, energy=energy)
        final.set_calculator(calc)
        new_struct.insert_structure(init_struct=init, final_struct=final)

reconfigure_almgsi()
