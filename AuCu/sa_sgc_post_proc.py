import dataset
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import UnivariateSpline

db_name = "data/sa_sgc_aucu.db"

def unique_chemical_pot():
    db = dataset.connect("sqlite:///{}".format(db_name))
    mu = []
    for row in db["results"].find():
        mu.append(row["mu_c1_0"])
    return np.unique(mu)

def heat_capacity():
    mu = unique_chemical_pot().tolist()
    db = dataset.connect("sqlite:///{}".format(db_name))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    for m in mu:
        sql = "SELECT temperature, heat_capacity FROM results WHERE "
        sql += "mu_c1_0 > {} AND mu_c1_0 < {}".format(m-0.001, m+0.001)
        T = []
        Cv = []
        for res in db.query(sql):
            T.append(res["temperature"])
            Cv.append(res["heat_capacity"])
            ax.plot(T, Cv)
    return fig


def show_cooling():
    mu = unique_chemical_pot().tolist()
    db = dataset.connect("sqlite:///{}".format(db_name))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cmap = mpl.cm.inferno
    norm = mpl.colors.Normalize(vmin=np.min(mu), vmax=np.max(mu))
    scalar_map = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array(mu)
    order_disorder_comp = []
    order_disorder_temp = []
    for m in mu:
        sql = "SELECT temperature, singlet_c1_0, heat_capacity FROM results WHERE "
        sql += "mu_c1_0 > {} AND mu_c1_0 < {}".format(m-0.001, m+0.001)
        T = []
        x = []
        Cv = []
        for res in db.query(sql):
            T.append(res["temperature"])
            x.append(res["singlet_c1_0"])
            Cv.append(res["heat_capacity"])
        indx_max = np.argmax(Cv)
        x = np.array(x)
        x = (1.0+x)/2.0
        if x[indx_max] > 0.25 and x[indx_max] < 0.8:
            order_disorder_comp.append(x[indx_max])
            order_disorder_temp.append(T[indx_max])
        ax.plot(x, T, color=scalar_map.to_rgba(m), lw=1)
    ax_divider = make_axes_locatable(ax)
    c_ax = ax_divider.append_axes("top", size="7%", pad="2%")
    cb = fig.colorbar(scalar_map, orientation="horizontal", cax=c_ax)
    cb.ax.xaxis.set_ticks_position('top')
    cb.ax.xaxis.set_label_position('top')
    cb.set_label("Chemical potential (eV)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Au concentration")
    ax.set_ylabel("Temperature (K)")

    sort_indx = np.argsort(order_disorder_comp)
    order_disorder_comp = np.array([order_disorder_comp[indx] for indx in sort_indx])
    order_disorder_temp = np.array([order_disorder_temp[indx] for indx in sort_indx])

    intervals = [0.25, 0.8]
    for i in range(len(intervals)-1):
        t_fit = order_disorder_temp[order_disorder_comp>=intervals[i]]
        x_fit = order_disorder_comp[order_disorder_comp>=intervals[i]]
        t_fit = t_fit[x_fit<=intervals[i+1]]
        x_fit = x_fit[x_fit<=intervals[i+1]]

        # Fit parabola
        coeff = np.polyfit(x_fit, t_fit, 9)
        poly = np.poly1d(coeff)
        x = np.linspace(x_fit[0], x_fit[-1], 100)
        ax.plot(x, poly(x), ls="-", color="grey", lw=3)
    ax.plot(order_disorder_comp, order_disorder_temp, ls="", marker="v", mfc="none")
    return fig

if __name__ == "__main__":
    from phase_diagram_aucu import plot
    plt.switch_backend("TkAgg")
    fig = show_cooling()
    plot(["AuCu3_AuCu", "AuCu_Au3Cu", "Au_Au3Cu", "AuCu3_Cu"],
         "data/phase_boundary_aucu.db", fig=fig, colors=["black"], lw=3, std_fill=False)
    # fig = heat_capacity()
    plt.show()
