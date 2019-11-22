from ase.db import connect
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
import os

SQLITE_DB = 'almgsiX_dft.db'
MYSQL_DB = os.environ['MYSQL_URL']

def sync():
    db_server = connect(MYSQL_DB)
    db_sqlite = connect(SQLITE_DB)

    num_inserted = 0
    symm_checker = SymmetryEquivalenceCheck()
    atoms_on_server = [r.toatoms() for r in db_server.select(type='initial')]
    groups = [row.get('group', -1) for row in db_sqlite.select()]
    for g in groups:
        rows = []
        for r in db_sqlite.select(group=g):
            rows.append(r)
        
        exists = False
        for r in rows:
            if symm_checker.compare(r.toatoms(), atoms_on_server):
                exists = True
                break
        
        if exists:
            continue

        for r in rows:
            try:
                row.__dict__.update({'project': 'cluster_expansion_almgsiX_fcc'})
                row._keys.append('project')
                #db_server.write(row)
                num_inserted += 1
            except Exception:
                pass
    print("Inserted {} calculations".format(num_inserted))

sync()