from ase.db import connect
import pandas as pd
import os
from ase.io.trajectory import TrajectoryWriter


data = pd.read_csv("data/bulk_mod_fit.csv")
db_local_name = "data/structures_with_bulk.db"
structures = []

traj = TrajectoryWriter("data/structure_with_bulk_mod.traj")
db_local = connect(db_local_name)
with connect(os.environ['MYSQL_URL']) as db:
    for g in data['Group']:
        print(g)
        try:
            row = db.get(group=g, struct_type='initial', project="cluster_expansion_almgsiX_fcc")
            db_local.write(row)
        except:
            pass

