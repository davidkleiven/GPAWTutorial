import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
import dataset
import numpy as np

db_name = "data/fixed_composition_order_param.db"

systems = {
    "QQ": "AuCu3",
    "PP": "AuCu",
    "NN": "Au3Cu"
}

expect_frac_swapped = {
    "QQ": 3.0/8.0,
    "PP": 1.0/2.0,
    "NN": 3.0/8.0
}


def plot():
    db = dataset.connect("sqlite:///{}".format(db_name))


    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    tbl = db["results"]
    for key, value in systems.items():
        T = []
        order = []
        for row in tbl.find(integration_path=key):
            T.append(row["temperature"])
            order.append(row["order_avg"])
        T = np.array(T)
        order = np.array(order)
        srt_indx = np.argsort(T)
        T = T[srt_indx]
        order = order[srt_indx]
        order = 1.0 - (order/1000.0)/expect_frac_swapped[key]

        ax.plot(T, order, marker="o", mfc="none", label=value)
    ax.set_xlabel("Temperature (K)")
    ax.set_ylabel("\$ 1 - f_\mathrm{diff}/f_\mathrm{diff,rnd}\$")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.axhline(0.0, ls="--", color="grey")
    ax.legend(frameon=False)
    plt.show()

if __name__ == "__main__":
    plot()
