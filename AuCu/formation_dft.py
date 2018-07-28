from ase.db import connect
db_name = "data/cu-au_fcc.db"
from matplotlib import pyplot as plt
import numpy as np

db = connect(db_name)
ref_energy_fcc = {
    "Au": -0.000,
    "Cu": -0.007
}

ref_energy_fcc = {
    "Au": -3.212,
    "Cu": -3.748
}

e_form = []
conc = []
ids = []
for row in db.select(converged=1):
    E = row["energy"]
    ids.append(row.id)
    natoms = row.natoms
    atoms = row.toatoms()
    count = {"Au": 0, "Cu": 0}
    symb = atoms.get_chemical_symbols()
    for s in symb:
        count[s] += 1
    if "Au" in count.keys():
        c_al = float(count["Au"])/natoms
    else:
        c_al = 0.0
    c_zn = 1.0 - c_al
    dE = E/natoms - ref_energy_fcc["Au"]*c_al - ref_energy_fcc["Cu"]*c_zn
    e_form.append(dE)
    conc.append(c_al)

min_indx = np.argmin(e_form)
print("Minimum ID: {}".format(ids[min_indx]))
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.plot(conc, e_form, ls="", marker="x")
ax.set_xlabel("Al concentration")
ax.set_ylabel("Energy of formation (eV/atom)")

plt.show()
