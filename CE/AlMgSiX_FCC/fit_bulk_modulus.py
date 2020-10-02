from ase.db import connect
from ase.eos import EquationOfState
from matplotlib import pyplot as plt

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
            eos = EquationOfState(volumes, energies, eos="murnaghan")
            v0, e0, B = eos.fit()
            e0, B, dBdP, V0 = eos.eos_parameters
            bulk_mod[g] = {
                'energy': e0,
                'bulk_mod': B,
                'volume': v0,
                'dBdP': dBdP
            }
            eos.plot(filename=f"fig/eos_fit/group{g}.png")
            plt.clf()
            plt.close()
        except Exception as exc:
            print(exc)

outfile = "data/bulk_mod_fit.csv"

with open(outfile, 'w') as out:
    out.write("Group,energy (eV),bulk_mod (eV/A^3),volume (A^3),dBdP\n")
    for k, v in bulk_mod.items():
        E = v['energy']
        B = v['bulk_mod']
        V = v['volume']
        dBdP = v['dBdP']
        out.write(f"{k},{E},{B},{V},{dBdP}\n")
