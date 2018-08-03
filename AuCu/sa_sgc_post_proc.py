import dataset
import numpy as np
import matplotlib as mpl
mpl.rcParams["svg.fonttype"] = "none"
mpl.rcParams["axes.unicode_minus"] = False
mpl.rcParams["font.size"] = 18
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from scipy.interpolate import UnivariateSpline
from cemc.tools import PeakExtractor
from scipy.signal import find_peaks_cwt

db_name = "data/sa_sgc_aucu_with_triplets.db"

tol = 1E-7
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
    ax = fig.add_subplot(1, 2, 1)
    for m in mu:
        sql = "SELECT temperature, sgc_heat_capacity, energy FROM results WHERE "
        sql += "mu_c1_0 > {} AND mu_c1_0 < {}".format(m-tol, m+tol)
        T = []
        Cv = []
        energy = []
        for res in db.query(sql):
            T.append(res["temperature"])
            energy.append(res["energy"])
            Cv.append(res["sgc_heat_capacity"])
        ax.plot(T, Cv)
    return fig

def heat_cap_from_energy(T, energy, mu, singlets):
    srt_indx = np.argsort(T)
    T = [T[indx] for indx in srt_indx]
    energy = [energy[indx] for indx in srt_indx]
    singlets = [singlets[indx] for indx in srt_indx]
    singlets = np.array(singlets)
    energy = np.array(energy)/1000.0
    T = np.array(T)
    energy -= mu*singlets
    spl = UnivariateSpline(T, energy, k=3)
    deriv = spl.derivative()
    # print(mu)
    # plt.plot(T, energy)
    # plt.plot(T, spl(T))
    # plt.show()
    return deriv(T)



def show_cooling():
    mu = unique_chemical_pot().tolist()
    db = dataset.connect("sqlite:///{}".format(db_name))
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    cmap = mpl.cm.copper
    norm = mpl.colors.Normalize(vmin=np.min(mu), vmax=np.max(mu))
    scalar_map = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    scalar_map.set_array(mu)
    order_disorder_comp = []
    order_disorder_temp = []
    mu_plot = []

    fig_T_mu = plt.figure()
    ax_T_mu = fig_T_mu.add_subplot(1, 1, 1)
    print("Number of unqitue chemical potentials: {}".format(len(mu)))
    for m in mu:
        sql = "SELECT temperature, singlet_c1_0, sgc_heat_capacity, energy FROM results WHERE "
        sql += "mu_c1_0 > {} AND mu_c1_0 < {} order by temperature".format(m-tol, m+tol)
        T = []
        x = []
        Cv = []
        U = []
        for res in db.query(sql):
            T.append(res["temperature"])
            x.append(res["singlet_c1_0"])
            Cv.append(res["sgc_heat_capacity"])
            U.append(res["energy"])

        peakind = find_peaks_cwt(Cv, np.arange(1, 40), noise_perc=10)

        x = np.array(x)
        x = (1.0+x)/2.0
        for indx_max in peakind:
            if x[indx_max] > 0.0 and x[indx_max] < 1.0:
                order_disorder_comp.append(x[indx_max])
                order_disorder_temp.append(T[indx_max])
                mu_plot.append(m)

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

    ax_T_mu.plot(mu_plot, order_disorder_temp, marker="o")

    sort_indx = np.argsort(order_disorder_comp)
    order_disorder_comp = np.array([order_disorder_comp[indx] for indx in sort_indx])
    order_disorder_temp = np.array([order_disorder_temp[indx] for indx in sort_indx])
    ax.plot(order_disorder_comp, order_disorder_temp, ls="", marker="v", mfc="none")


    intervals = [0.25, 0.8]
    for i in range(len(intervals)-1):
        t_fit = order_disorder_temp[order_disorder_comp>=intervals[i]]
        x_fit = order_disorder_comp[order_disorder_comp>=intervals[i]]
        t_fit = t_fit[x_fit<=intervals[i+1]]
        x_fit = x_fit[x_fit<=intervals[i+1]]

        # Fit parabola
        # coeff = np.polyfit(x_fit, t_fit, 9)
        # poly = np.poly1d(coeff)
        # x = np.linspace(x_fit[0], x_fit[-1], 100)
        # ax.plot(x, poly(x), ls="-", color="grey", lw=3)
    return fig

def plot_experimental(fig):
    ax = fig.get_axes()[0]

    data = np.loadtxt("data/experimental_phase_diag.csv", delimiter=",")
    conc = 1.0 - data[:, 0]
    T = data[:, 1] + 273.15
    ax.plot(conc, T)
    return fig

if __name__ == "__main__":
    from phase_diagram_aucu import plot
    plt.switch_backend("TkAgg")
    fig = show_cooling()
    # plot(["AuCu3_AuCu", "AuCu_Au3Cu", "Au_Au3Cu", "AuCu3_Cu"],
    #      "data/phase_boundary_aucu.db", fig=fig, colors=["black"], lw=3, std_fill=False)
    fig = heat_capacity()
    # fig = plot_experimental(fig)
    plt.show()
