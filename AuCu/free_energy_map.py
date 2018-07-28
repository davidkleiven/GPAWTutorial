from cemc.tools import FreeEnergy
import numpy as np
import dataset
from matplotlib import pyplot as plt
plt.switch_backend("TkAgg")

db_name = "data/sgc_aucu.db"


def get_unique_chemical_potentials():
    db = dataset.connect("sqlite:///{}".format(db_name))
    chem_pot = []
    tbl = db["results"]
    for row in tbl.find():
        chem_pot.append(row["mu_c1_0"])
    return np.unique(chem_pot)


def get_unique_temperatures():
    db = dataset.connect("sqlite:///{}".format(db_name))
    T = []
    tbl = db["results"]
    for row in tbl.find():
        T.append(row["temperature"])
    return np.unique(T)


def update_db(phases):
    mu = get_unique_chemical_potentials().tolist()
    print(mu)

    db = dataset.connect("sqlite:///{}".format(db_name))
    tbl = db["results"]
    for phase in phases:
        for chem in mu:
            sql = "SELECT id, energy, singlet_c1_0, temperature FROM results "
            sql += "WHERE init_formula=\"{}\" AND mu_c1_0 > {} AND ".format(phase, chem-0.001)
            sql += "mu_c1_0 < {}".format(chem+0.001)

            U = []
            T = []
            singlet = []
            for res in db.query(sql):
                U.append(res["energy"]/1000.0)
                T.append(res["temperature"])
                singlet.append(res["singlet_c1_0"])

            if not T:
                continue
            print (mu)
            free = FreeEnergy(limit="lte")
            chem_pot = {"c1_0": chem}
            singl = {"c1_0": singlet}
            sgc_energy = free.get_sgc_energy(U, singl, chem_pot)
            res = free.free_energy_isochemical(T=T, sgc_energy=sgc_energy, nelem=2)

            # Put the result into the database
            for T, phi in zip(res["temperature"], res["free_energy"]):
                sql = "SELECT id FROM results WHERE mu_c1_0 > {} AND mu_c1_0 < {} AND temperature={}".format(chem-0.001, chem+0.001, T)
                res = db.query(sql)
                uid = res.next()["id"]
                new_cols = {"id": uid, "free_energy": phi}
                tbl.update(new_cols, ["id"])


def create_plot_map(phases):
    ls = ["-", "--", "-."]
    unique_temps = get_unique_temperatures().tolist()

    db = dataset.connect("sqlite:///{}".format(db_name))
    tbl = db["results"]
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for i, phase in enumerate(phases):
        mu = []
        phi = []
        for T in unique_temps:
            for res in tbl.find(temperature=T, init_formula=phase):
                mu.append(res["mu_c1_0"])
                phi.append(res["free_energy"])
            srt_indx = np.argsort(mu)
            mu = [mu[indx] for indx in srt_indx]
            phi = [phi[indx] for indx in srt_indx]
            ax.plot(mu, phi, ls=ls[i], marker="o")
    plt.show()


if __name__ == "__main__":
    # update_db(["Au250Cu750", "Au500Cu500"])
    create_plot_map(["Au250Cu750"])
