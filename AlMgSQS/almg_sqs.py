from ase.build import bulk
from icet import ClusterSpace
from icet.tools.structure_generation import generate_sqs
from ase.io import write
from ase.db import connect
from icet.io.logging import set_log_config
set_log_config(level='INFO')

DB_NAME = 'sqs_db.db'
def get_sqs(conc):
    atoms = bulk("Al", a=4.05)
    cs = ClusterSpace(atoms, [5.0, 5.0], ['Al', 'Mg'])

    sqs = generate_sqs(cluster_space=cs, max_size=16, target_concentrations=conc, include_smaller_cells=False)
    return sqs

def get_sqs_hcp(conc):
    atoms = bulk('Mg')
    cs = ClusterSpace(atoms, [10.0, 7.0], [['Al', 'Mg'], ['O', 'H']])
    conc = {'Al': 0.25, 'Mg': 0.25, 'O': 0.25, 'H': 0.25}
    sqs = generate_sqs(cluster_space=cs, max_size=16, target_concentrations=conc, include_smaller_cells=False)
    return sqs

def generate_all_sqs():
    db = connect(DB_NAME)
    for n in range(2, 16, 2):
        conc = {'Al': n/16.0, 'Mg': 1.0 - n/16.0}
        print(conc)
        sqs = get_sqs(conc=conc, crystalstructure=lattice)
        db.write(sqs, lattice='fcc')

def generate_all_sqs_hcp():
    db = connect(DB_NAME)
    for n in range(2, 16, 2):
        conc = {'Al': n/16.0, 'Mg': 1.0 - n/16.0}
        sqs = get_sqs_hcp(conc)
        db.write(sqs, lattice='hcp')

generate_all_sqs_hcp()


