from ase.db import connect
from matplotlib import pyplot as plt
import dataset

ref_cu = -3.748
ref_au = -3.212

print(-28.28/8 - ref_cu*4/8 - ref_au*4/8)
print(-26.92/8 - ref_cu*2/8 - ref_au*6/8)

def energy_form():
    db = connect("cu-au_quad.db")
    x = []
    form = []
    for row in db.select(struct_type="final"):
        e = row.energy
        num_atoms = row.count_atoms()
        if "Cu" not in num_atoms.keys():
            num_atoms["Cu"] = 0
        if "Au" not in num_atoms.keys():
            num_atoms["Au"] = 0

        dE = e - ref_cu*num_atoms["Cu"]/row.natoms - ref_au*num_atoms["Au"]/row.natoms
        x.append(num_atoms["Cu"]/row.natoms)
        form.append(dE)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, form, marker="o")
    plt.show()

def energy_vs_temp():
    db = dataset.connect("sqlite:///cu-au.db")
    tbl = db["results"]
    e = []
    t = []
    for row in tbl.find(Au_conc=0.5):
        e.append(row["energy"])
        t.append(row["temperature"])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t, e, "o", mfc="none")
    plt.show()
if __name__ == "__main__":
    #energy_form()
    energy_vs_temp()