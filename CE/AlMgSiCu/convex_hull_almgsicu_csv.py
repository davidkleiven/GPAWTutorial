import matplotlib as mpl
mpl.rcParams.update({'font.size': 11, 'font.family': 'serif'})
from clease.basis_function import Polynomial
import numpy as np
from clease.tools import singlets2conc
from matplotlib import pyplot as plt
from scipy.spatial import ConvexHull

bf = Polynomial(['Al', 'Mg', 'Si', 'Cu'])

data_file = "data/almgsicu.csv"
ce_file = "data/almgsicu_predictions.csv"
ce_file_fic = "data/almgsicu_predictions_fic.csv"
fig_exts = ["png", "svg", "pdf"]

REF_ENG = {
    "Al": -3.73715645818146, 
    "Mg": -1.60775899474433,  
    "Si": -4.800166410712106,
    "Cu": -3.67403582995655
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

def ce_data(fic=False):
    fname = ce_file
    if fic:
        fname = ce_file_fic
    try:
        value, err = np.loadtxt(fname, delimiter=",", skiprows=1, unpack=True)
    except Exception as exc:
        print(exc)
        return None, None
    return value, err

def convex(mg_conc, si_conc, cu_conc, formation):
    pts = np.zeros((len(mg_conc)+2, 4))
    pts[:-2, 0] = mg_conc
    pts[:-2, 1] = si_conc
    pts[:-2, 2] = cu_conc
    pts[:-2, 3] = formation

    # Add pure Si
    pts[-2, 0] = 0.0
    pts[-2, 1] = 1.0
    pts[-2, 2] = 0.0

    # Visibility option
    pts[-1, 0] = 0.25
    pts[-1, 1] = 0.25
    pts[-1, 2] = 0.25
    pts[-1, 3] = -200.0
    qhull = ConvexHull(pts, qhull_options=f"QG{pts.shape[0]-1}")

    on_convex = set()
    for simplex in qhull.simplices[qhull.good]:
        for s in simplex:
            if s < len(mg_conc):
                on_convex.add(s)
    return on_convex

def plot_convex_structs(dft_cnv, dft, on_convex_ce=None, on_convex_ce_fic=None):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)

    x = [dft['Mg'][i] for i in range(0, len(dft['Mg']))]
    x_dft = [x[i] for i in dft_cnv]
    E_dft = [dft['formation'][i] for i in dft_cnv]
    ax.plot(x_dft, E_dft, 'o', mfc='none', color="#333333", label="DFT")

    if on_convex_ce is not None:
        both = set(dft_cnv).intersection(on_convex_ce)
        print(f"Fraction agree {ce_file}: {len(both)/len(dft_cnv)}")
        x_both = [x[i] for i in both]
        E_both = [dft['formation'][i] for i in both]
        ax.plot(x_both, E_both, 'o', color="#333333", label="DFT+AICC")

        only_ce = set(on_convex_ce).difference(dft_cnv)
        x_ce = [x[i] for i in only_ce]
        E_ce = [dft['formation'][i] for i in only_ce]
        print("ONLY CE AICC:", only_ce)
        ax.plot(x_ce, E_ce, 'v', mfc='none', color="#333333", label="AICC")

    if on_convex_ce_fic is not None:
        both = set(dft_cnv).intersection(on_convex_ce_fic)
        print(f"Fraction agree {ce_file_fic}: {len(both)/len(dft_cnv)}")
        x_both = [x[i] for i in both]
        E_both = [dft['formation'][i] for i in both]
        ax.plot(x_both, E_both, 'x', mfc='none', color="#0bd526", label="DFT+FIC")

        only_ce = set(on_convex_ce_fic).difference(dft_cnv)
        print("ONLY FIC:", only_ce)
        x_ce = [x[i] for i in only_ce]
        E_ce = [dft['formation'][i] for i in only_ce]
        ax.plot(x_ce, E_ce, '^', mfc='none', color="#0bd526", label="FIC")

    symbs = ['Al', 'Mg', 'Si', 'Cu']
    for j, i in enumerate(dft_cnv):
        formula = ''
        formula = {}
        
        minval = min(dft[s][i] for s in symbs if dft[s][i] > 1e-3)
        for s in symbs:
            num = int(np.round(dft[s][i]/minval, decimals=0))
            if num > 1e-3:
                formula[s] = num
        
        fstring = ''
        for k in symbs:
            v = formula.get(k, 0)
            if v == 1:
                fstring += k
            elif v > 1:
                fstring += f"{k}$_{v}$"
        ax.annotate(fstring, (x_dft[j]+0.02, E_dft[j]+0.001), fontsize=8)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Mg concentration")
    ax.set_ylabel("DFT Formation energy (eV/atom)")
    ax.legend(loc="best")
    fig.tight_layout()

def plot_convex():
    dft = dft_data()
    ce, err = ce_data()
    if ce is not None:
        ce -= sum(REF_ENG[k]*np.array(v) for k, v in dft.items() if k in REF_ENG.keys())

    ce_fic, err = ce_data(fic=True)
    if ce_fic is not None:
        ce_fic -= sum(REF_ENG[k]*np.array(v) for k, v in dft.items() if k in REF_ENG.keys())

    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    x_conc = 'Mg'
    ax.plot(dft[x_conc], dft['formation'], 'o', mfc='none', color="#333333")
    for i, xy in enumerate(zip(dft[x_conc], dft['formation'])):
        ax.annotate(str(i), xy)
    if ce is not None:
        ax.errorbar(dft[x_conc], ce, yerr=err, fmt='.', capsize=2, color='#6290DFCC')
    ax.set_xlabel(f"{x_conc} concentration")
    ax.set_ylabel("Formation Energy (eV/atom)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    on_convex = convex(dft['Mg'], dft['Si'], dft['Cu'], dft['formation'])
    print("On convex: ", on_convex)

    on_convex_ce = None
    if ce is not None:
        on_convex_ce = convex(dft['Mg'], dft['Si'], dft['Cu'], ce)

    on_convex_fic = None
    if ce is not None:
        on_convex_fic = convex(dft['Mg'], dft['Si'], dft['Cu'], ce_fic)
    plot_convex_structs(on_convex, dft, on_convex_ce, on_convex_fic)
    
    # for i in on_convex:
    #     print(f"{i}. Mg: {dft['Mg'][i]} Al: {dft['Al'][i]} Si: {dft['Si'][i]} Cu: {dft['Cu'][i]}")
    fig.tight_layout()
    
    for ext in fig_exts:
        fig.savefig(f"data/{fig_name()}.{ext}", dpi=300)

plot_convex()
plt.show()
