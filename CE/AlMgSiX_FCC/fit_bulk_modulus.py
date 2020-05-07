from ase.db import connect
from ase.eos import EquationOfState

db_name = "almgsiX_murnaghan.db"

groups = set()
with connect(db_name) as db:
    for row in db.select():
        groups.add(row.group)

bulk_mod = {}

with connect(db_name) as db:
    for g in groups:
        volumes = []
        energies = []
        for row in db.select(group=g):
            volumes.append(row.volume)
            energies.append(row.energy)
        try:
            eos = EquationOfState(volumes, energies, eos="birchmurnaghan")
            v0, e0, B = eos.fit()
            bulk_mod[g] = {
                'energy': e0,
                'bulk_mod': B,
                'volume': v0
            }
        except:
            pass

outfile = "data/bulk_mod_fit.csv"

with open(outfile, 'w') as out:
    out.write("Group,energy (eV),bulk_mod (eV/A^3),volume (A^3)\n")
    for k, v in bulk_mod.items():
        E = v['energy']
        B = v['bulk_mod']
        V = v['volume']
        out.write(f"{k},{E},{B},{V}\n")
