import dataset
import matplotlib as mpl
from cemc.tools import CanonicalFreeEnergy
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
from ase.units import kJ, mol
plt.switch_backend("TkAgg")

ref_energy_fcc = {
    "Al": -30.029/8.0,
    "Zn": -8.605/8.0
}

ref_energy_hcp = {
    "Al": -59.414/16.0,
    "Zn": -17.820/16.0
}

def free_energy_of_formation(db_name, ref_energy):
    """Computes the free energy of formation."""
    db = dataset.connect("sqlite:///{}".format(db_name))
    tbl = db["results"]
    temps = [800, 700, 600, 500, 400, 300, 200, 100]
    concs = np.arange(0.05, 1, 0.05)
    all_F = {}
    c_found = []
    for c in concs:
        statement = "SELECT temperature, energy FROM results WHERE al_conc > {} AND al_conc < {}".format(c-0.01, c+0.01)
        T = []
        U = []
        for res in db.query(statement):
            T.append(res["temperature"])
            U.append(res["energy"]/1000.0)
        comp = {"Al": c, "Zn": 1-c}
        if not T:
            continue
        c_found.append(c)
        free_eng = CanonicalFreeEnergy(comp)
        T, U, F = free_eng.get(T, U)
        for temp, f in zip(list(T), list(F)):
            if temp not in all_F.keys():
                all_F[temp] = [f]
            else:
                all_F[temp].append(f)
    cmap = mpl.cm.copper
    norm = mpl.colors.Normalize(vmin=np.min(temps), vmax=np.max(temps))
    scalar_map = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array(temps)

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    concs = np.array(c_found)
    for T in temps:
        f_form = np.array(all_F[T]) - concs * ref_energy["Al"] - (1.0-concs)*ref_energy["Zn"]
        f_form *= mol/kJ
        f_form = f_form.tolist()
        f_form.insert(0, 0)
        f_form.append(0)
        ax.plot([0.0]+concs.tolist()+[1.0], f_form, color=scalar_map.to_rgba(T), marker="^")
    ax.set_xlabel("Al concentration")
    ax.set_ylabel("Free energy of formation (kJ/mol)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax_divider = make_axes_locatable(ax)
    c_ax = ax_divider.append_axes("top", size="7%", pad="2%")
    cb = fig.colorbar(scalar_map, orientation="horizontal", cax=c_ax)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.set_label("Temperature (K)")
    plt.show()


def enthalpy_of_formation(db_name, ref_energy):
    db = dataset.connect("sqlite:///{}".format(db_name))
    tbl = db["results"]
    temps = [800, 700, 600, 500, 400, 300, 200, 100]

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cmap = mpl.cm.copper
    norm = mpl.colors.Normalize(vmin=np.min(temps), vmax=np.max(temps))
    scalar_map = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array(temps)
    for T in temps:
        concs = []
        e_form = []
        for row in tbl.find(temperature=T):
            energy = row["energy"]/1000.0
            conc = row["al_conc"]
            dE = energy - ref_energy["Al"]*conc - ref_energy["Zn"]*(1.0 - conc)
            concs.append(conc)
            e_form.append(dE)
        concs += [0.0, 1.0]
        e_form += [0.0, 0.0]
        srt_indx = np.argsort(concs)
        concs = np.array(concs)
        e_form = np.array(e_form)
        e_form = e_form[srt_indx]
        concs = concs[srt_indx]
        ax.plot(concs, e_form*mol/kJ, color=scalar_map.to_rgba(T), marker="o")

    ax_divider = make_axes_locatable(ax)
    c_ax = ax_divider.append_axes("top", size="7%", pad="2%")
    cb = fig.colorbar(scalar_map, orientation="horizontal", cax=c_ax)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.set_label("Temperature (K)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_xlabel("Al concentration")
    ax.set_ylabel("Enthalpy of formation (kJ/mol)")
    #ax.legend(frameon=False)
    plt.show()


if __name__ == "__main__":
    # enthalpy_of_formation("data/sa_alzn_fcc_run2.db", ref_energy_fcc)
    # free_energy_of_formation("data/sa_alzn_fcc_run2.db", ref_energy_fcc)
    # enthalpy_of_formation("data/sa_alzn_hcp.db", ref_energy_hcp)
    free_energy_of_formation("data/sa_alzn_hcp.db", ref_energy_hcp)
