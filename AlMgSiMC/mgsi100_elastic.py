from atomtools.ase import ElasticConstants
from ase.io import read

DB_NAME = "elastic_mgsi100.db"

def prepare_mgsi():
    atoms = read("data/mgsi100_fully_relaxed.xyz")
    elastic = ElasticConstants(atoms, DB_NAME)
    elastic.prepare_db()

if __name__ == "__main__":
    prepare_mgsi()