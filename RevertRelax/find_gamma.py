from ase.db import connect
import os
db_name = os.environ['MYSQL_URL']

with connect(db_name) as db:
    for row in db.select():
        if row.natoms == 58:
            print(row.id)