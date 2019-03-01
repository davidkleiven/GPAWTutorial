from ase.db import connect
import matplotlib as mpl
mpl.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 18})
from matplotlib import pyplot as plt
import numpy as np
from ase.units import kJ
from ase.units import m as meter
from scipy.optimize import curve_fit

ref_energies = {
    "Al": -3.737,
    "MgSi": -218.272/64.0
}


def func(x, const, pref, damping):
    return const + pref*np.exp(-damping*x)


def main():
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    surface_energies = [[], [], []]
    natoms = [[], [], []]
    layer_thickness = [[], [], []]
    db = connect("data/surface_tension_mgsi.db")
    for surf in range(2):
        for row in db.select([('id', '>=', 47), ('id', "<=", 59)]):
            init_id = row.init_id
            init_row = db.get(id=init_id)
            surf = init_row.surface
            E = row.energy/row.natoms
            if surf <= 1:
                E -= 0.5*(ref_energies["Al"] + ref_energies["MgSi"])
            else:
                E -= ref_energies["MgSi"]

            natoms[surf].append(row.natoms)
            lengths = row.toatoms().get_cell_lengths_and_angles()[:3]
            length = np.max(lengths)
            layer_thickness[surf].append(length/2.0)
            area = 2*np.prod(lengths)/np.max(lengths)
            gamma = E/area
            gamma *= 1000.0*1.6022*10.0
            surface_energies[surf].append(gamma)

    labels = ["Al-MgSi alternating", "Al-MgSi Single", "MgSi - MgSi"]
    i = 0
    fit_x = np.linspace(3.0, 25.0, 100)
    colors = ["#3182bd", "#636363", "#a6bddb"]
    for n, E in zip(layer_thickness, surface_energies):
        indx = np.argsort(n)
        n = [n[x] for x in indx]
        E = [E[x] for x in indx]
        params, pcov = curve_fit(func, n, E, p0=[5.0, 2.0, 0.4])
        print(params[0])
        fitted = func(fit_x, params[0], params[1], params[2])
        ax.plot(n, E, "o", mfc="none", label=labels[i], color=colors[i])
        ax.plot(fit_x, fitted, color=colors[i])
        ax.axhline(params[0], ls="--", color=colors[i])
        i += 1
    ax.legend(frameon=False)
    ax.set_ylabel("Surface tension (mJ/m\$^2\$)")
    ax.set_xlabel("Layer thickness (\AA)")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    plt.show()

if __name__ == "__main__":
    main()