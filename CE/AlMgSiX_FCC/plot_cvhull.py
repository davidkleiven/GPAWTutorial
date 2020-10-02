import dataset
import matplotlib as mpl
mpl.rcParams.update({'font.family': 'serif', 'font.size': 11})
from matplotlib import pyplot as plt

REF_ENG = {
    "Al": -3.735496408086985, 
    "Mg": -1.2858192465708942, 
    "Si": -4.800166410712106,
    "X": 0.0
}

N = 256
def main():
    db = dataset.connect("sqlite:///data/almgsi_mc_sgc.db")
    tbl = db['sa_fixed_conc']

    al_concs = []
    si_concs = []
    mg_concs = []
    energies = []
    for row in tbl:
        E = row['energy']/N
        c_Al = row['Al_conc']
        c_Mg = row['Mg_conc']
        c_Si = row['Si_conc']
        c_Al = c_Al if c_Al is not None else 0.0
        c_Mg = c_Mg if c_Mg is not None else 0.0
        c_Si = c_Si if c_Si is not None else 0.0
        dE = E - REF_ENG['Al']*c_Al - REF_ENG['Mg']*c_Mg - REF_ENG['Si']*c_Si
        energies.append(dE)
        al_concs.append(c_Al)
        mg_concs.append(c_Mg)
        si_concs.append(c_Si)


    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(mg_concs, energies, 'o', mfc='none', alpha=0.2, color='#111111')
    ax.set_xlabel("Mg concentration")
    ax.set_ylabel("Formation energy (eV/atom)")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    fig.tight_layout()
    fig.savefig("chull.png", dpi=300)

main()