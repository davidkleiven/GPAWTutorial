import dataset
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["font.size"] = 18
mpl.rcParams["axes.unicode_minus"] = False
from matplotlib import pyplot as plt

def main():
    db_name = "sqlite:///data/effective_concentration.db"
    db = dataset.connect(db_name)
    temperatures = [200,300,400,500,600,700,800]

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    syst = db["systems"]
    conc_tbl = db["concentration"]
    eff_conc_tbl = db["effective_conc"]
    act_table = db["activity_coefficient"]
    for T in temperatures:
        entries = syst.find(temperature=T)
        sysIDs = [entry["id"] for entry in entries]
        mg_concs = []
        eff_mg_concs = []
        for uid in sysIDs:
            N = 1000
            conc_entry = conc_tbl.find_one(sysID=uid)
            x_al = conc_entry["Al"]
            x_mg = conc_entry["Mg"]
            mg_concs.append(x_mg)
            norm = x_al*N/(1.0+x_mg*N)
            act = act_table.find_one(sysID=uid)["AltoMg"]*norm
            eff_mg_conc = (N-act)/(1+act)
            eff_mg_concs.append(eff_mg_conc/N)

        ax.plot( mg_concs, eff_mg_concs, marker="^", mfc="none", label="{}K".format(int(T)))

    ax.legend( loc="best", frameon=False )
    x = np.linspace(0.0,1.0,10)
    ax.plot(x, x, "--", lw=3, color="grey")
    ax.set_xlabel("Mg concentration")
    ax.set_ylabel("Effective Mg concentration")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()
