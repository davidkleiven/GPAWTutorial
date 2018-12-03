import dataset
import numpy as np
import matplotlib as mpl
from cemc.tools import CanonicalFreeEnergy
import json

mpl.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 18})

DB_NAME_CLUSTER = ["sqlite:///data/heat_almg_cluster_size8.db",
                   "sqlite:///data/heat_almg_cluster_size12.db",
                   "sqlite:///data/heat_almg_cluster_size18.db",
                   "sqlite:///data/heat_almg_cluster_size36.db"]

size = {
    "sqlite:///data/heat_almg_cluster_size8.db": 8,
    "sqlite:///data/heat_almg_cluster_size12.db": 12,
    "sqlite:///data/heat_almg_cluster_size18.db": 18,
    "sqlite:///data/heat_almg_cluster_size36.db": 36
}

DB_NAME_PURE = "sqlite:///data/pure_al3mg3.db"

def free_energy_vs_size(temps=[293, 353]):
    db_name = "sqlite:///data/heat_almg_cluster_size_all_size.db"
    outfname = "data/almg_review/free_energy_size.json"
    sizes = range(3, 51)
    db = dataset.connect(db_name)
    tbl = db["thermodynamic"]
    free_energies = {int(T): [] for T in temps}
    entropy = {int(T): [] for T in temps}
    internal_energy = {int(T): [] for T in temps}
    for size in sizes:
        energy = []
        temperature = []
        for row in tbl.find(size=size):
            energy.append(row["energy"])
            temperature.append(row["temperature"])
        energy = np.array(energy)
        temperature = np.array(temperature)
        indx = np.argsort(temperature)
        temperature = temperature[indx]
        energy = energy[indx]
        comp = {"Al": 0.75, "Mg": 0.25}
        free_eng = CanonicalFreeEnergy(comp, limit="lte")
        T, U, F = free_eng.get(temperature, energy)
        S = (U-F)/T
        for target_temp in temps:
            indx = np.argmin(np.abs(target_temp-T))
            free_energies[target_temp].append(F[indx])
            internal_energy[target_temp].append(U[indx])
            entropy[target_temp].append(S[indx])

    data = {}
    data["sizes"] = list(sizes)
    data["free_energy"] = free_energies
    data["entropy"] = entropy
    data["internal_energy"] = internal_energy
    # for target_temp in temps:
    #     free_energies[target_temp] = list(free_energies[target_temp])

    with open(outfname, 'w') as outfile:
        json.dump(data, outfile)
    print("Free energies written to {}".format(outfname))

def pure_phase():
    from matplotlib import pyplot as plt
    db = dataset.connect(DB_NAME_PURE)
    energy = []
    temperature = []
    order_param = []
    tbl = db["thermodynamic"]
    for row in tbl.find():
        energy.append(row["energy"])
        temperature.append(row["temperature"])
        order_param.append(row["site_order"])
    comp = {"Al": 0.75, "Mg": 0.25}
    free_eng = CanonicalFreeEnergy(comp, limit="lte")
    
    T, U, F = free_eng.get(temperature, energy)
    S = (U-F)/T

    F *= (4.0/1000.0) # Convert to per f.u.
    S *= (4.0/1000.0) # Convert to per f.u.

    # Plot the order parameter
    f = 1.0 - (np.array(order_param)/1000.0)*8.0/3.0
    fig_order = plt.figure()
    ax_order = fig_order.add_subplot(1, 1, 1)
    ax_order.plot(temperature, f, marker="o", mfc="none", color="#5D5C61")
    ax_order.set_xlabel("Temperature (K)")
    ax_order.set_ylabel("Order parameter")
    ax_order.spines["right"].set_visible(False)
    ax_order.spines["top"].set_visible(False)

    # Plot free energy
    fig_free = plt.figure()
    ax_free = fig_free.add_subplot(1, 1, 1)
    ax_free.plot(T, (F-F[0])*1000.0, marker="o", mfc="none", color="#557A95")
    ax_entropy = ax_free.twinx()
    ax_entropy.plot(T, S*1000.0, marker="v", mfc="none", color="#5D5C61")
    #ax_free.axvline(x=293, ls="--", color="#B1A296")
    ax_free.set_xlabel("Temperature (K)")
    ax_free.set_ylabel("Free energy change (meV/f.u.)")
    ax_entropy.set_ylabel("Entropy (meV/f.u./K)")

def cluster():
    from matplotlib import pyplot as plt
    fig_free = plt.figure()
    ax_free = fig_free.add_subplot(1, 1, 1)
    ax_entropy = ax_free.twinx()
    colors = ["#5D5C61", "#379683", "#7395AE", "#557A95", "#B1A296"]
    for col, db_name in zip(colors, DB_NAME_CLUSTER):
        db = dataset.connect(db_name)
        energy = []
        temperature = []
        tbl = db["thermodynamic"]
        for row in tbl.find():
            energy.append(row["energy"])
            temperature.append(row["temperature"])
        comp = {"Al": 0.75, "Mg": 0.25}
        free_eng = CanonicalFreeEnergy(comp, limit="lte")
        
        T, U, F = free_eng.get(temperature, energy)

        S = (U-F)/T

        # Plot free energy
        ax_free.plot(T, F-F[0], marker="o", mfc="none", color=col, label=size[db_name])
        ax_entropy.plot(T, S*1000.0, marker="v", mfc="none", color=col)
    ax_free.set_xlabel("Temperature (K)")
    ax_free.set_ylabel("Free energy change (eV)")
    ax_free.legend(loc="best", frameon=False)
    ax_entropy.set_ylabel("Entropy (meV/K)")

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    #pure_phase()
    #cluster()
    #plt.show()

    temps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90,
         100, 150, 200, 220, 240, 260, 280, 300, 320, 340,  360, 380,
         400, 420, 440, 460, 480, 500]

    free_energy_vs_size(temps=temps)