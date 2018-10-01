from ase.db import connect
db_name = "data/zn-al_fcc.db"
db_name_hcp = "data/zn-al_hcp.db"
from matplotlib import pyplot as plt
import numpy as np

db = connect(db_name)
ref_energy_fcc = {
    "Al": -30.029/8.0,
    "Zn": -8.605/8.0
}

e_form = []
conc = []
ids = []
for row in db.select(converged=1):
    E = row["energy"]
    ids.append(row.id)
    natoms = row.natoms
    atoms = row.toatoms()
    count = {"Al": 0, "Zn": 0}
    symb = atoms.get_chemical_symbols()
    for s in symb:
        count[s] += 1
    if "Al" in count.keys():
        c_al = float(count["Al"])/natoms
    else:
        c_al = 0.0
    c_zn = 1.0 - c_al
    dE = E/natoms - ref_energy_fcc["Al"]*c_al - ref_energy_fcc["Zn"]*c_zn
    e_form.append(dE)
    conc.append(c_al)

min_indx = np.argmin(e_form)
print("Minimum ID: {}".format(ids[min_indx]))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(conc, e_form, ls="", marker="x")
ax.set_xlabel("Al concentration")
ax.set_ylabel("Energy of formation (eV/atom)")

e_form = []
conc = []
ids = []
db = connect(db_name_hcp)
for row in db.select(converged=1):
    E = row["energy"]
    ids.append(row.id)
    natoms = row.natoms
    atoms = row.toatoms()
    count = {"Al": 0, "Zn": 0}
    symb = atoms.get_chemical_symbols()
    for s in symb:
        count[s] += 1
    if "Al" in count.keys():
        c_al = float(count["Al"])/natoms
    else:
        c_al = 0.0
    c_zn = 1.0 - c_al
    dE = E/natoms - ref_energy_fcc["Al"]*c_al - ref_energy_fcc["Zn"]*c_zn
    e_form.append(dE)
    conc.append(c_al)

ax.plot(conc, e_form, ls="", marker="o")

plt.show()
