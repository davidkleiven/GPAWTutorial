from ase.db import connect

db = connect("almgsi_newconfig.db")
workdir = "data/almgsi_gpw_files"
for row in db.select(converged=1):
    num_entries = 0
    for entry in db.select(name=row.name):
        num_entries += 1
    print(num_entries)
