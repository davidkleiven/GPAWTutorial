import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
from clease.basis_function import Polynomial
import numpy as np
from clease.tools import singlets2conc
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

bf = Polynomial(['Al', 'Mg', 'Si', 'Cu'])

data_file = "data/almgsicu.csv"
ce_file = "data/almgsi_train_predictions.csv"
#ce_file = "data/almgsi_train_fic_predictions.csv"
fig_exts = ["png", "svg", "pdf"]

REF_ENG = {
    "Al": -3.735496408086985, 
    "Mg": -1.608,  
    "Si": -4.800166410712106,
    "Cu": -3.674
}

def fig_name():
    dft_prefix = data_file.split("/")[-1][:-4]
    ce_prefix = ce_file.split("/")[-1][:-4]
    return dft_prefix + "_" + ce_prefix

def dft_data():
    data = np.loadtxt(data_file, delimiter=',')
    singlets = data[:, 1:4]
    energy = data[:, -1]
    conc = singlets2conc(bf.get_basis_functions(), singlets)

    res = {
        'Mg': [],
        'Si': [],
        'Al': [],
        'Cu': [],
        'formation': []
    }

    for i, item in enumerate(conc):
        res['Mg'].append(item['Mg'])
        res['Si'].append(item['Si'])
        res['Al'].append(item['Al'])
        res['Cu'].append(item['Cu'])
        res['formation'].append(energy[i] - sum(REF_ENG[k]*v for k, v in item.items()))
    return res

def ce_data():
    try:
        value, err = np.loadtxt(ce_file, delimiter=",", skiprows=1, unpack=True)
    except Exception as exc:
        print(exc)
        return None, None
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
    if ce is not None:
        ce -= sum(REF_ENG[k]*np.array(v) for k, v in dft.items() if k in REF_ENG.keys())

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(dft['Mg'], dft['formation'], 'o', mfc='none', color="#333333")
    # for i, xy in enumerate(zip(dft['Mg'], dft['formation'])):
    #     ax.annotate(str(i), xy)
    if ce is not None:
        ax.errorbar(dft['Mg'], ce, yerr=err, fmt='.', capsize=2, color='#6290DFCC')
    ax.set_xlabel("Mg concentration")
    ax.set_ylabel("Formation Energy (eV/atom)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    on_convex = convex(dft['Mg'], dft['Si'], dft['formation'])
    
    for i in on_convex:
        print(f"{i}. Mg: {dft['Mg'][i]} Al: {dft['Al'][i]} Si: {dft['Si'][i]} Cu: {dft['Cu'][i]}")
    fig.tight_layout()
    
    for ext in fig_exts:
        fig.savefig(f"data/{fig_name()}.{ext}", dpi=300)

plot_convex()
plt.show()
