import dataset
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import numpy as np
from ase.units import kJ, mol

ref_energy_fcc = {
    "Au": -3.212,
    "Cu": -3.748
}



def enthalpy_of_formation(db_name, ref_energy, field):
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
            if field == "energy":
                energy = row[field]/1000.0
            elif field == "entropy":
                energy = row[field]*1000.0
            else:
                energy = row[field]
            conc = row["au_conc"]
            dE = energy - ref_energy["Au"]*conc - ref_energy["Cu"]*(1.0 - conc)
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
    ax.set_xlabel("Au concentration")
    if field == "energy":
        ax.set_ylabel("Enthalpy of formation (kJ/mol)")
    elif field == "entropy":
        ax.set_ylabel("Entropy of formation (J/K mol)")
    elif field == "free_energy":
        ax.set_ylabel("Free energy of formation (kJ/mol)")

    #ax.legend(frameon=False)


if __name__ == "__main__":
    ref_entropy = {"Au": 0.0, "Cu": 0.0}
    enthalpy_of_formation("data/sa_aucu_only_pairs.db", ref_energy_fcc, "energy")
    enthalpy_of_formation("data/sa_aucu_only_pairs.db", ref_energy_fcc, "free_energy")
    enthalpy_of_formation("data/sa_aucu_only_pairs.db", ref_entropy, "entropy")
    plt.show()
    # enthalpy_of_formation("data/sa_alzn_hcp.db", ref_energy_hcp)
