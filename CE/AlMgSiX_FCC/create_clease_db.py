from ase.db import connect
from ase.io.trajectory import TrajectoryWriter
import os
import re

mysql_url = os.environ['MYSQL_URL']
relax_type = 'volume_relax'
project = 'cluster_expansion_almgsiX_fcc'

def to_traj():
    traj_init = 'structures_initial.traj'
    traj_final = 'structures_final.traj'

    if os.path.exists(traj_init):
        os.remove(traj_init)

    if os.path.exists(traj_final):
        os.remove(traj_final)

    init = TrajectoryWriter(traj_init)
    final = TrajectoryWriter(traj_final)

    db_server = connect(mysql_url)
    groups = set()
    for row in db_server.select(project=project):
        gr = row.get('group', None)

        if gr is not None:
            groups.add(gr)

    for gr in groups:
        print(gr)
        try:
            atoms_init = list(db_server.select(group=gr, type='initial', project=project))[0].toatoms()
            atoms_final = list(db_server.select(group=gr, type=relax_type, project=project))[0].toatoms()
            init.write(atoms_init)
            final.write(atoms_final)
        except (KeyError, IndexError):
            pass
    init.close()
    final.close()


to_traj()
