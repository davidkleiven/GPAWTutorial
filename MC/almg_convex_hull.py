from ase.db import connect
from scipy.spatial import ConvexHull
import numpy as np
from ase.units import kJ, mol
import matplotlib as mpl
mpl.rcParams.update({"axes.unicode_minus": False, "svg.fonttype": "none", "font.size": 18})

def conc_and_energies(db_name, ref_al, ref_mg):
    db = connect(db_name)
    energies = []
    conc = []
    for row in db.select(converged=True):
        energies.append(row.energy/row.natoms)
        count = row.count_atoms()
        if "Al" not in count.keys():
            count["Al"] = 0.0
        if "Mg" not in count.keys():
            count["Mg"] = 0.0
        for k in count.keys():
            count[k] /= float(row.natoms)
        conc.append(count)

    enthalp_form = []
    mg_conc = []
    for c, eng in zip(conc, energies):
        dE = eng - ref_al*c["Al"]- ref_mg*c["Mg"]
        enthalp_form.append(dE*mol/kJ)
        mg_conc.append(c["Mg"])
    return mg_conc, enthalp_form

def main():
    from matplotlib import pyplot as plt
    db_name_fcc = "../CE/ce_hydrostatic.db"
    ref_al = -3.7366703865074444 
    ref_mg = -1.5909140339677044
    mg_conc, enthalp_form = conc_and_energies(db_name_fcc, ref_al, ref_mg)

    pts = np.vstack((mg_conc, enthalp_form)).T
    qhull = ConvexHull(pts)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Gamma phase structures
    mg_conc_217, enthalp_form_217 = conc_and_energies("../CE/almg_217.db", ref_al, ref_mg)
    ax.plot(mg_conc_217, enthalp_form_217, "o", color="black", mfc="#557A95", label="\$\gamma\$")

    for simplex in qhull.simplices:
        x1 = pts[simplex[0], 0]
        x2 = pts[simplex[1], 0]
        y1 = pts[simplex[0], 1]
        y2 = pts[simplex[1], 1]
        if y1 > 0.0 or y2 > 0.0:
            continue
        ax.plot([x1, x2], [y1, y2], "--", color="#7395AE")
    ax.plot(mg_conc, enthalp_form, "o", mfc="#5D5C61", color="black", label="FCC")

    # HCP structures
    mg_conc_hcp, enthalp_form_hcp = conc_and_energies("data/updated.db", ref_al, ref_mg)
    ax.plot(mg_conc_hcp, enthalp_form_hcp, "o", color="black", mfc="#B1A296", label="HCP")
    ax.legend()
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.set_ylabel("Formation energies (kJ/mol)")
    ax.set_xlabel("Mg concentration")
    plt.show()


if __name__ == "__main__":
    main()