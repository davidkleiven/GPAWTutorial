from cemc.tools import DatasetAverager
import h5py
import dataset
import glob
import numpy as np
import matplotlib as mpl
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
plt.switch_backend("TkAgg")

db_name = "phase_boundary_aucu.db"

Tmax = {
    "AuCu3_Cu": 1000,
    "AuCu3_AuCu": 650,
    "AuCu_Au3Cu": 700,
    "Au_Au3Cu": 1000
}

def average_results(boundary):
    folder = "data/{}_decrease/".format(boundary)
    Tmin = 100000
    Tmax = 0
    all_dsets = []
    for fname in glob.glob(folder+"phase_boundary*.h5"):
        with h5py.File(fname, 'r') as hfile:
            singlets = np.array(hfile["boundary1/singlets"])
            mu = np.array(hfile["boundary1/chem_pot"])
            temperature = np.array(hfile["boundary1/temperatures"])
            new_min_values = np.min(singlets, axis=0)
            new_max_values = np.max(singlets, axis=0)
            dset = {
                "singlets": singlets,
                "temperature": temperature,
                "mu": mu
            }
            all_dsets.append(dset)
            if np.max(temperature) > Tmax:
                Tmax = np.max(temperature)
            if np.min(temperature) < Tmin:
                Tmin = np.min(temperature)

    T = np.linspace(Tmin, Tmax, 100)
    line1 = DatasetAverager(T)
    line2 = DatasetAverager(T)
    mu_avg = DatasetAverager(T)
    for dset in all_dsets:
        line1.add_dataset(dset["temperature"], dset["singlets"][:, 0].T[0])
        line2.add_dataset(dset["temperature"], dset["singlets"][:, 1].T[0])
        mu_avg.add_dataset(dset["temperature"].tolist(), dset["mu"].T[0])

    field_dict = {
        "x_values": "temperature",
        "y_values": "singlet",
        "std_y": "singlet_std"
    }
    db_name = "data/phase_boundary_aucu_high_to_low.db"
    info = {"boundary": boundary}
    line1.save_to_db(
        db_name=db_name, table="composition_temperature",
        fields=field_dict, name="first", info=info)
    line2.save_to_db(
        db_name=db_name, table="composition_temperature",
        fields=field_dict, name="second", info=info)
    field_dict = {
        "x_values": "temperature",
        "y_values": "chemical_potential",
        "std_y": "chemical_potential_std"
        }
    mu_avg.save_to_db(
        db_name=db_name, table="mu_temperature",
        fields=field_dict, info=info)


def plot(boundaries, db_name, fig=None, colors=None, lw=2, std_fill=True):
    fig_provided = False
    if fig is not None:
        ax = fig.get_axes()[0]
        fig_provided = True
    else:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
    db = dataset.connect("sqlite:///{}".format(db_name))
    tbl = db["composition_temperature"]
    if colors is None:
        colors = ['#66c2a5','#fc8d62','#8da0cb','#e78ac3','#a6d854','#ffd92f']
        
    for num, b in enumerate(boundaries):
        for tag in ["first", "second"]:
            conc = []
            T = []
            std = []
            for res in tbl.find(boundary=b, name=tag):
                if res["temperature"] < Tmax[b]:
                    conc.append(res["singlet"])
                    T.append(res["temperature"])
                    std.append(res["singlet_std"])
            if not T:
                continue

            srt_indx = np.argsort(T)
            T = [T[indx] for indx in srt_indx]
            conc = [conc[indx] for indx in srt_indx]
            conc = np.array(conc)
            conc += 1.0
            conc /= 2.0
            ax.plot(conc, T, color=colors[num%len(colors)], lw=lw)
            conc_minus = np.array(conc) - np.array(std)
            conc_plus = np.array(conc) + np.array(std)
            if std_fill:
                ax.fill_betweenx(T, conc_minus, conc_plus, color="#d9d9d9")
    if not fig_provided:
        ax.set_xlabel("Au concentration")
        ax.set_ylabel("Temperature (K)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
    else:
        fig.canvas.draw()
        fig.canvas.flush_events()
    return fig


if __name__ == "__main__":
    # average_results("AuCu3_Cu")
    # plot(["AuCu3_Cu"], "data/phase_boundary_aucu_high_to_low.db")
    plot(["AuCu3_AuCu", "AuCu_Au3Cu", "Au_Au3Cu", "AuCu3_Cu"], "data/phase_boundary_aucu.db")
    plt.show()
