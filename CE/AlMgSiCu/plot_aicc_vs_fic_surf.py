import matplotlib as mpl
mpl.rcParams.update({'font.family': 'serif', 'font.size': 11})
from matplotlib import pyplot as plt
from scipy.stats import linregress
import numpy as np

PT_COLOR = "#6290DF"
LINE_COLOR = "#333333"

def get_data(data_file):
    dsets = ['pure', 'alt']
    series = ['inv_size', 'dft', 'aicc', 'fic']
    res = {k: {s: [] for s in series} for k in dsets}

    data_type = 'pure'
    keys = ['inv_size', 'dft', 'aicc', 'fic']
    with open(data_file) as infile:
        for line in infile:
            if line.startswith('Inv'):
                if 'pure' in line.lower():
                    data_type = 'pure'
                elif 'alt' in line.lower():
                    data_type = 'alt'
                else:
                    raise ValueError("Could not deduce data type")
                continue

            nums = [float(x) for x in line.strip().split(',')]
            for k, n in zip(keys, nums):
                res[data_type][k].append(n)
    return res

def get_mfc(dtype):
    if dtype == 'pure':
        return 'none'
    return PT_COLOR

def plot(data):
    fig = plt.figure(figsize=(4, 3))
    ax = fig.add_subplot(1, 1, 1)
    markers = {'dft': 'v', 'aicc': '^', 'fic': 'd'}
    ls = {'dft': '-', 'aicc': '--', 'fic': '-.'}
    x = np.array([0.01, 0.13])
    for dtype in ['pure', 'alt']:
        dset = data[dtype]
        mfc = get_mfc(dtype)
        for series, m in markers.items():
            lw = 1
            if series == 'dft':
                lw = 2
            slope, interscept, _, _, _ = linregress(dset['inv_size'], dset[series])
            ax.plot(x, interscept + slope*x, '--', color=LINE_COLOR, lw=lw, ls=ls[series])
            label = f'{series}-{dtype}'
            ax.plot(dset['inv_size'], dset[series], m, mfc=mfc, color=PT_COLOR, label=label)
            print(f"Type {dtype}. Series: {series}. Slope: {slope}")
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Inverse size (1/N)")
    ax.set_ylabel("Energy per atom (eV/atom)")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("data/aicc_vs_fic_surf.pdf")
    fig.savefig("data/aicc_vs_fic_surf.svg")

def main():
    data = get_data("data/aicc_vs_fic_surface_dataset.csv")
    plot(data)
    plt.show()

main()