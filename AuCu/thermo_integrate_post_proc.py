import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import savgol_filter
import dataset
from scipy.interpolate import splprep, splev

sgc_db_name = "data/sa_sgc_aucu_thermo_integrate.db"

transition_points = {
    "A": {"point": [390], "type": "temperature"},
    "B": {"point": [0.217, 0.227], "type": "mu"},
    "C": {"point": [0.217, 0.227], "type": "mu"},
    "D": {"point": [0.200, 238], "type": "mu"}
}

temp_paths = ["A"]

def get_sharp_change(x, y):
    w_length = len(y)/8
    if w_length%2 == 0:
        w_length += 1
    second_der = savgol_filter(y, window_length=w_length, polyorder=2, deriv=2)
    max_der = np.max(np.abs(second_der))
    peaks = np.where(np.abs(second_der) > 0.8 * max_der)
    return x[peaks], y[peaks], second_der[peaks], peaks

def get_path(path_id):
    db = dataset.connect("sqlite:///{}".format(sgc_db_name))
    tbl = db["results"]
    T = []
    order = []
    comp = []
    mu = []
    for res in tbl.find(integration_path=path_id):
        T.append(res["temperature"])
        order.append(res["order_avg"])
        comp.append(res["singlet_c1_0"])
        mu.append(res["mu_c1_0"])
    return np.array(T), np.array(order), np.array(mu), np.array(comp)


def detect_transition_point(x, y, show=False):
    """Detect a transition point by linear fit."""
    transition_indx = 0
    for max_include in range(3, len(y)-1):
        mean = np.mean(y[:max_include])
        mean = y[0]
        std = np.std(y[:max_include])
        if abs(y[max_include] - mean) > 50.0:
            transition_indx = max_include - 1
            break
    return transition_indx

def show_integration_path(path_id, x_value):

    T, order, mu, comp = get_path(path_id)
    if x_value == "temperature":
        sort_indx = np.argsort(T)
    else:
        sort_indx = np.argsort(mu)
    T = T[sort_indx]
    order = order[sort_indx]
    mu = mu[sort_indx]
    comp = comp[sort_indx]

    if x_value == "temperature":
        x_peak, order_peak, deriv, peaks = get_sharp_change(T, order)
    else:
        x_peak, order_peak, deriv, peaks = get_sharp_change(mu, order)
    print(x_peak, order_peak, deriv)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    if x_value == "temperature":
            ax.plot( T, order )
    else:
        ax.plot( mu, order )
    ax.plot(x_peak, order_peak, ls="", marker="o")
    ax.set_xlabel(x_value)
    return fig

def plot_phase_stability(paths, fig):
    ax = fig.get_axes()[0]
    points = []
    for path in paths:
        T, order, mu, comp = get_path(path)
        if order[-1] < order[0]:
            order = order[::-1]
            T = T[::-1]
            mu = mu[::-1]
            comp = comp[::-1]
        comp_trans = []
        T_trans = []
        if path in temp_paths:
            indx = detect_transition_point(mu, order, show=False)
        else:
            indx = detect_transition_point(T, order, show=False)
        comp_trans.append(comp[indx])
        T_trans.append(T[indx])
        print(path, T[indx], comp[indx])

        for c, T in zip(comp_trans, T_trans):
            points.append([c, T])

    points = np.array(points)
    ax.plot(points[:, 0], points[:, 1], ls="--", marker="o")

    # Fit parabola through curve
    indx_min = np.argmin(points[:, 0])
    indx_max = np.argmax(points[:, 0])

def main():
    fig = plt.figure()
    fig.add_subplot(1, 1, 1)
    # show_integration_path("A", "temperature")
    show_integration_path("OO", "mu")
    paths = ["C", "QQ", "MM", "OO", "P", "A", "PP", "O", "M", "Q", "CC"]
    plot_phase_stability(paths, fig)
    plt.show()

if __name__ == "__main__":
    main()
