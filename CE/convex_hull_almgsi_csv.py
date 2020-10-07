from clease.basis_function import Polynomial
import numpy as np
from clease.tools import singlets2conc
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

bf = Polynomial(['Al', 'Mg', 'Si'])

data_file = "data/almgsi.csv"
ce_file = "data/almgsi_predictions.csv"

REF_ENG = {
    "Al": -3.735496408086985, 
    "Mg": -1.608,  
    "Si": -4.800166410712106,
    "Cu": -3.674
}

def dft_data():
    data = np.loadtxt(data_file, delimiter=',')
    singlets = data[:, 1:3]
    energy = data[:, -1]
    conc = singlets2conc(bf.get_basis_functions(), singlets)

    res = {
        'Mg': [],
        'Si': [],
        'Al': [],
        'formation': []
    }

    for i, item in enumerate(conc):
        res['Mg'].append(item['Mg'])
        res['Si'].append(item['Si'])
        res['Al'].append(item['Al'])
        res['formation'].append(energy[i] - sum(REF_ENG[k]*v for k, v in item.items()))
    return res

def ce_data():
    value, err = np.loadtxt(ce_file, delimiter=",", skiprows=1, unpack=True)
    return value, err

def convex(mg_conc, si_conc, formation):
    pts = np.zeros((len(mg_conc), 3))
    pts[:, 0] = mg_conc
    pts[:, 1] = si_conc
    pts[:, 2] = formation
    qhull = ConvexHull(pts)

    tol = 0.001
    on_convex = set()
    for simplex in qhull.simplices:
        for s in simplex:
            if formation[s] < tol:
                on_convex.add(s)
    return on_convex

def plot_convex():
    dft = dft_data()
    ce, err = ce_data()
    ce -= sum(REF_ENG[k]*np.array(v) for k, v in dft.items() if k in REF_ENG.keys())

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dft['Mg'], dft['formation'], 'o', mfc='none')
    ax.errorbar(dft['Mg'], ce, fmt='.', capsize=2)

    on_convex = convex(dft['Mg'], dft['Si'], dft['formation'])
    
    for i in on_convex:
        print(f"{i}. Mg: {dft['Mg'][i]} Al: {dft['Al'][i]} Si: {dft['Si'][i]}")

plot_convex()
plt.show()
