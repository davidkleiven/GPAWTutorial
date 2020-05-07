from ase.db import connect
import os

def get_data():
    local_db = 'almgsiX_murnaghan.db'
    db_server = os.environ['MYSQL_URL']

    atom_rows = []
    with connect(db_server) as db:
        for row in db.select(project='almgsiX_murnaghan'):
            print(f"Extracting row. {row.id}")
            atom_rows.append(row)

    with connect(local_db) as db:
        for row in atom_rows:
            print(f"Inserting row {row.id}")
            db.write(row)

get_data()