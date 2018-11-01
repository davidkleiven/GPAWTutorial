from atomtools.ase import ElasticConstants
from ase.io import read
from ase.build import bulk
from ase.visualize import view

DB_NAME = "al3mg_elastic.db"

def prepare():
    atoms = bulk("Al")*(2, 2, 2)
    atoms[4].symbol = "Mg"
    atoms[3].symbol = "Mg"
    factor = 0.5*8.284/4.05
    cell = atoms.get_cell()
    cell *= factor
    atoms.set_cell(cell, scale_atoms=True)
    elastic = ElasticConstants(atoms, DB_NAME)
    elastic.prepare_db()

if __name__ == "__main__":
    prepare()