import dataset
import numpy as np
import matplotlib as mpl
from cemc.tools import CanonicalFreeEnergy

mpl.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 18})

DB_NAME_CLUSTER = "sqlite:///data/heat_almg_cluster3.db"
DB_NAME_PURE = "sqlite:///data/pure_al3mg3.db"

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
    ax_free.plot(T, F-F[0], marker="o", mfc="none", color="#557A95")
    ax_entropy = ax_free.twinx()
    ax_entropy.plot(T, S*1000.0, marker="v", mfc="none", color="#5D5C61")
    ax_free.set_xlabel("Temperature (K)")
    ax_free.set_ylabel("Free energy change (eV)")
    ax_entropy.set_ylabel("Entropy (meV/K)")

def cluster():
    from matplotlib import pyplot as plt
    db = dataset.connect(DB_NAME_CLUSTER)
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
    fig_free = plt.figure()
    ax_free = fig_free.add_subplot(1, 1, 1)
    ax_free.plot(T, F-F[0], marker="o", mfc="none", color="#557A95")
    ax_entropy = ax_free.twinx()
    ax_entropy.plot(T, S*1000.0, marker="v", mfc="none", color="#5D5C61")
    ax_free.set_xlabel("Temperature (K)")
    ax_free.set_ylabel("Free energy change (eV)")
    ax_entropy.set_ylabel("Entropy (meV/K)")

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    pure_phase()
    cluster()
    plt.show()
    