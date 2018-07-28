import gpaw as gp
from ase.io import read
from ase.db import connect
from atomtools.eos import BirschMurnagan
from ase.units import GPa
import numpy as np
from matplotlib import pyplot as plt

run = False
db_name = "mgsi_bulkmod.db"

if run:
    atoms = read("data/relaxed_mgsi.xyz")

    calc = gp.GPAW(mode=gp.PW(600), xc="PBE", kpts=(4, 4, 4), nbands=-100)
    atoms.set_calculator(calc)

    scale = [1.04, 1.06, 1.08]

    db = connect(db_name)
    for s in scale:
        atoms_sim = atoms.copy()
        atoms_sim.set_cell(atoms.get_cell()*s, scale_atoms=True)
        atoms_sim.set_calculator(calc)
        energy = atoms_sim.get_potential_energy()
        db.write(atoms_sim, key_value_pairs={"scale": s})
else:
    db = connect(db_name)
    V = []
    E = []
    for row in db.select():
        V.append(row.volume)
        E.append(row.energy)

    V = np.array(V)
    E = np.array(E)
    eos = BirschMurnagan(V, E)
    eos.plot()
    E0, V0 = eos.minimum_energy()
    B = eos.bulk_modulus(V0)
    print("Bulk modulus: {} GPa".format(B/GPa))
    print("Minimum energy: {} eV".format(E0))
    print("Minimum volume: {}".format(V0))
    plt.show()
