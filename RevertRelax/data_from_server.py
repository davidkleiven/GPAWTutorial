from ase.db import connect
from ase.io.trajectory import TrajectoryWriter
import os

projects = ["ce_prebeta_interstitial"]#cluster_expansion_almgsiX_fcc"]
            #"cluster_expansion_fesi_bcc"]

db_server_name = os.environ['MYSQL_URL']


with connect(db_server_name) as db:
    for p in projects:
        initial = {}
        final = {}
        for row in db.select(project=p):
            str_type = row.get('struct_type', '')
            if str_type == '':
                str_type = row.get('type', '')
            name = row.get('name', '') + f"{row.get('group', 0)}"
            if str_type == 'initial':
                initial[name] = row.toatoms()
            elif str_type in ['final', 'full_relax', 'full_relax_gpw15']:
                final[name] = row.toatoms()

        images = []
        for n in initial.keys():
            atoms1 = initial[n]
            atoms2 = final.get(n, None)
            if atoms2 is not None:
                images.append(atoms1)
                images.append(atoms2)
        
        fname = f"data/{p}.traj"
        writer = TrajectoryWriter(fname, atoms=images)
        for img in images:
            writer.write(img)
        writer.close()
        print(f"Wrote {len(images)/2} pairs for project {p}")