import matplotlib as mpl
mpl.rcParams.update({'font.family': 'serif', 'font.size': 11})
from matplotlib import pyplot as plt
from ase.db import connect
from collections import Counter
import numpy as np
from scipy.spatial import ConvexHull

# REF_EMT_ENERGIES = {
#     'Al': -0.005,
#     'Cu': -0.007
# }

REF_EMT_ENERGIES = {
    'Ag': 0.0,
    'Pt': 0.0
}

REF_EMT_ENERGIES = {
    'Cu': -0.007,
    'Pd': 0.0
}


def emt_data():
    db = connect("data/cupd_ce.db")
    concs = []
    form_energies = []
    for row in db.select([('struct_type', '=', 'final')]):
        energy = row.energy
        atoms = row.toatoms()
        counter = Counter(atoms.symbols)

        conc1 = counter.get('Cu', 0.0)/len(atoms)
        conc2 = counter.get('Pd', 0.0)/len(atoms)
        dE = energy/len(atoms) - conc1*REF_EMT_ENERGIES['Cu'] - conc2*REF_EMT_ENERGIES['Pd']
        concs.append(conc1)
        form_energies.append(dE)
    return concs, form_energies

def ce_data():
    X = np.loadtxt("data/cupd.csv", delimiter=',', skiprows=1)
    singlet = X[:, 1]
    conc = 0.5*(1+singlet)
    pred, std = np.loadtxt("data/predictions_fic.csv", delimiter=',', skiprows=1, unpack=True)
    pred -= (conc*REF_EMT_ENERGIES['Cu'] + (1.0-conc)*REF_EMT_ENERGIES['Pd'])
    return conc, pred, std

def dft_convex(concs, form_energies):
    pts = np.zeros((len(concs), 2))
    pts[:, 0] = concs
    pts[:, 1] = form_energies
    hull = ConvexHull(pts)
    simplices = [s for s in hull.simplices if form_energies[s[0]] < 0.007 and form_energies[s[1]] < 0.007]
    return simplices

def std_vs_conc():
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    conc, _, std = ce_data()

    ax.plot(conc, std*1000.0, 'v', mfc='none', color='#6290DF')
    ax.set_ylabel("Standard error (meV/atom)")
    ax.set_xlabel("Cu concentration")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig("data/std_error.pdf")
    fig.savefig("data/std_error.svg")
    fig.savefig("data/std_error.png", dpi=200)

def formation_energy():
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    al_conc_emt, dE_emt = emt_data()

    conc, dE_ce, std = ce_data()
    ax.errorbar(conc, dE_ce, yerr=2*std, fmt='.', capsize=2, markersize=5, color='#6290DFCC')
    ax.plot(al_conc_emt, dE_emt, 'o', mfc='none', color='black')
    simplex = dft_convex(al_conc_emt, dE_emt)
    for s in simplex:
        ax.plot([al_conc_emt[s[0]], al_conc_emt[s[1]]], [dE_emt[s[0]], dE_emt[s[1]]], color="#333333")
    stable = list(set([x for s in simplex for x in s]))
    print(stable)
    print([al_conc_emt[s] for s in stable])
    print(set([d for x in [list(x) for x in dft_convex(conc, dE_ce)] for d in x]))
    ax.set_xlabel("Cu concentration")
    ax.set_ylabel("Formation energy (eV/atom)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig("data/formation_energy.pdf")
    fig.savefig("data/formation_energy.svg")
    fig.savefig("data/formation_energy.png", dpi=200)

formation_energy()
std_vs_conc()
plt.show()