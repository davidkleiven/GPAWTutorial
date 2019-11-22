from ase.db import connect
from ase.utils.structure_comparator import SymmetryEquivalenceCheck
import os
import sys

SQLITE_DB = 'almgsiX_dft.db'
MYSQL_DB = os.environ['MYSQL_URL']

def sync(group):
    db_server = connect(MYSQL_DB)
    db_sqlite = connect(SQLITE_DB)

    num_inserted = 0
    symm_checker = SymmetryEquivalenceCheck()
    atoms_on_server = [r.toatoms() for r in db_server.select(type='initial')]
    rows = []
    for r in db_sqlite.select(group=group):
        rows.append(r)
        
    for r in rows:
        if symm_checker.compare(r.toatoms(), atoms_on_server):
            exists = True
            print("Structures already exists in DB") 
            return

    for row in rows:
        if row.get('struct_type', 'none') == 'initial':
            row.__dict__.update({'type': 'initial'})
            if 'type' not in row._keys:
                row._keys.append('type')
        try:
            row.__dict__.update({'project': 'cluster_expansion_almgsiX_fcc'})
            row._keys.append('project')
            db_server.write(row)
            num_inserted += 1
        except Exception:
            pass
    print("Inserted {} calculations".format(num_inserted))

sync(int(sys.argv[1]))
